//! Audio-reactive shaders: capture the default sink's monitor over PipeWire,
//! run a short FFT, and publish a [`AudioUniforms`] snapshot the render path
//! uploads to the shader's spectrum UBO each frame.
//!
//! A single global capture serves every output — the audio is the system's
//! output mix, identical on all monitors. The PipeWire client runs its own
//! mainloop on a dedicated thread and is spun up lazily, only once a shader
//! references the audio uniforms (see [`crate::app`]). When PipeWire is
//! absent or the graph can't be captured, the snapshot stays zeroed and
//! shaders simply render silence.
//!
//! The FFT/binning approach (Hann window, log-spaced bins, asymmetric
//! attack/release smoothing) follows the spectrum meter in the sibling
//! `damascene-volume` project.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use bytemuck::Zeroable;
use pipewire as pw;
use pw::spa;
use pw::{properties::properties, spa::pod::Pod};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;

use crate::gpu::{AudioUniforms, AUDIO_BINS};

/// FFT window length and hop (in samples). 2048 @ 48 kHz ≈ 43 ms of context,
/// recomputed every 1024 samples (~21 ms) — smooth enough for a wallpaper
/// without chasing every transient.
const FFT_SIZE: usize = 2048;
const FFT_HOP: usize = 1024;
/// Spectrum range. Below ~35 Hz is mostly rumble; above ~18 kHz is inaudible
/// to most and noisy in monitor captures.
const MIN_HZ: f32 = 35.0;
const MAX_HZ: f32 = 18_000.0;

/// A running PipeWire capture of the default sink's monitor. Holds the worker
/// thread and the shared snapshot it writes; dropping it asks the thread to
/// quit and joins.
pub struct AudioCapture {
    shared: Arc<Mutex<AudioUniforms>>,
    /// Sending wakes the capture's mainloop and asks it to quit — works even
    /// in silence, when no `process` callback would otherwise fire.
    quit: pw::channel::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl AudioCapture {
    /// Spawn the capture thread. Returns immediately; the snapshot stays
    /// zeroed until audio flows (and forever if capture can't start).
    pub fn start() -> AudioCapture {
        let shared = Arc::new(Mutex::new(AudioUniforms::zeroed()));
        let (quit, quit_rx) = pw::channel::channel();
        let thread_shared = shared.clone();
        let thread = thread::Builder::new()
            .name("prism-bg-audio".into())
            .spawn(move || {
                if let Err(e) = run(thread_shared, quit_rx) {
                    tracing::warn!("audio capture unavailable: {e}; shaders render silence");
                }
            })
            .expect("spawn audio capture thread");
        AudioCapture {
            shared,
            quit,
            thread: Some(thread),
        }
    }

    /// The latest spectrum, for upload to a shader's UBO. Cheap (a 144-byte
    /// copy under a briefly-held lock).
    pub fn snapshot(&self) -> AudioUniforms {
        self.shared
            .lock()
            .map(|g| *g)
            .unwrap_or_else(|_| AudioUniforms::zeroed())
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        // Ask the mainloop to quit; ignore the error if the thread already
        // exited (e.g. capture never started).
        let _ = self.quit.send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Per-stream userdata for the capture listener.
struct CaptureData {
    shared: Arc<Mutex<AudioUniforms>>,
    /// Negotiated audio format (channels + rate), filled on `param_changed`.
    format: spa::param::audio::AudioInfoRaw,
    /// Built once the sample rate is known.
    processor: Option<SpectrumProcessor>,
}

/// Run the PipeWire mainloop: connect a capture stream to the default sink's
/// monitor and pump its buffers through the FFT until asked to quit.
fn run(
    shared: Arc<Mutex<AudioUniforms>>,
    quit_rx: pw::channel::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    // Clean shutdown: quit the loop when AudioCapture is dropped.
    let quit_loop = mainloop.clone();
    let _quit_recv = quit_rx.attach(mainloop.loop_(), move |_| quit_loop.quit());

    // `stream.capture.sink=true` makes WirePlumber connect us to the *default
    // sink's monitor* and follow it as the default changes — exactly the
    // "react to whatever's playing" behaviour a wallpaper wants. DSP role +
    // our node name keep monitoring tools from metering us back.
    let props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "DSP",
        *pw::keys::NODE_NAME => "prism-bg.spectrum",
        "stream.capture.sink" => "true",
    };

    let stream = pw::stream::StreamBox::new(&core, "prism-bg-spectrum", props)?;
    let data = CaptureData {
        shared,
        format: Default::default(),
        processor: None,
    };

    let _listener = stream
        .add_local_listener_with_user_data(data)
        .state_changed(|_, data, _, state| {
            if let pw::stream::StreamState::Error(err) = state {
                tracing::warn!("audio capture stream error: {err}");
                data.processor = None;
            }
        })
        .param_changed(|_, data, id, param| {
            let Some(param) = param else { return };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Ok((media_type, media_subtype)) = format_utils::parse_format(param) else {
                return;
            };
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            let _ = data.format.parse(param);
        })
        .process(process)
        .register()?;

    // Ask for 32-bit float, stereo (PipeWire down/up-mixes the sink to match);
    // the exact rate is negotiated and read back in `param_changed`.
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    audio_info.set_channels(2);
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )?
    .0
    .into_inner();
    let mut params = [Pod::from_bytes(&values).ok_or("failed to build PipeWire format pod")?];

    stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    mainloop.run();
    Ok(())
}

/// `process` callback: deinterleave one buffer to mono, feed the FFT, and
/// publish the spectrum if a new column came out.
fn process(stream: &pw::stream::Stream, data: &mut CaptureData) {
    let Some(mut buffer) = stream.dequeue_buffer() else {
        return;
    };
    let datas = buffer.datas_mut();
    if datas.is_empty() {
        return;
    }
    let channels = data.format.channels() as usize;
    let rate = data.format.rate().max(1);
    let chunk_bytes = datas[0].chunk().size() as usize;
    let Some(samples) = datas[0].data() else {
        return;
    };
    let bytes = chunk_bytes.min(samples.len());
    if channels == 0 || bytes < 4 {
        return;
    }

    // Interleaved f32 frames → mono (channel average).
    let frames = (bytes / 4) / channels;
    if frames == 0 {
        return;
    }
    let mut mono = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut sum = 0.0_f32;
        for ch in 0..channels {
            let start = (frame * channels + ch) * 4;
            if let Ok(b) = samples[start..start + 4].try_into() {
                sum += f32::from_le_bytes(b);
            }
        }
        mono.push(sum / channels as f32);
    }

    let processor = data
        .processor
        .get_or_insert_with(|| SpectrumProcessor::new(rate));
    if processor.rate != rate {
        *processor = SpectrumProcessor::new(rate);
    }

    if let Some(bins) = processor.push(&mono) {
        let uniforms = pack(&bins);
        if let Ok(mut g) = data.shared.lock() {
            *g = uniforms;
        }
    }
}

/// Pack `AUDIO_BINS` normalized magnitudes into the std140 UBO layout, with
/// the overall level and three band summaries (low/mid/high thirds).
fn pack(bins: &[f32; AUDIO_BINS]) -> AudioUniforms {
    let mut u = AudioUniforms::zeroed();
    for (i, &b) in bins.iter().enumerate() {
        u.bins[i / 4][i % 4] = b;
    }
    let third = AUDIO_BINS / 3;
    let mean = |s: &[f32]| s.iter().copied().sum::<f32>() / s.len().max(1) as f32;
    u.bass = mean(&bins[..third]);
    u.mid = mean(&bins[third..2 * third]);
    u.treble = mean(&bins[2 * third..]);
    u.level = mean(bins);
    u
}

/// Sliding-window FFT that turns a mono stream into log-spaced, smoothed
/// spectrum bins in `0..1`.
struct SpectrumProcessor {
    rate: u32,
    window: Vec<f32>,
    pending: Vec<f32>,
    scratch: Vec<Complex<f32>>,
    fft: Arc<dyn Fft<f32>>,
    /// Previous bins, for attack/release smoothing across columns.
    last: [f32; AUDIO_BINS],
}

impl SpectrumProcessor {
    fn new(rate: u32) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        // Hann window to cut spectral leakage.
        let window = (0..FFT_SIZE)
            .map(|i| {
                let phase = std::f32::consts::TAU * i as f32 / (FFT_SIZE - 1) as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        SpectrumProcessor {
            rate,
            window,
            pending: Vec::with_capacity(FFT_SIZE * 2),
            scratch: vec![Complex::default(); FFT_SIZE],
            fft,
            last: [0.0; AUDIO_BINS],
        }
    }

    /// Append samples and return the most recent spectrum column once at least
    /// one full window is available (older windows in the same call are
    /// dropped — a wallpaper only needs the latest).
    fn push(&mut self, mono: &[f32]) -> Option<[f32; AUDIO_BINS]> {
        self.pending.extend_from_slice(mono);
        let mut latest = None;
        while self.pending.len() >= FFT_SIZE {
            latest = Some(self.column());
            self.pending.drain(..FFT_HOP);
        }
        latest
    }

    fn column(&mut self) -> [f32; AUDIO_BINS] {
        for (i, c) in self.scratch.iter_mut().enumerate() {
            c.re = self.pending[i] * self.window[i];
            c.im = 0.0;
        }
        self.fft.process(&mut self.scratch);

        let nyquist = self.rate as f32 * 0.5;
        let max_hz = MAX_HZ.min(nyquist);
        let (min_log, max_log) = (MIN_HZ.ln(), max_hz.ln());
        let hz_to_index = |hz: f32| ((hz / self.rate as f32) * FFT_SIZE as f32).round() as usize;

        let mut bins = [0.0_f32; AUDIO_BINS];
        for (bin, out) in bins.iter_mut().enumerate() {
            // Log-spaced band edges, so bass detail isn't crushed.
            let lo_t = bin as f32 / AUDIO_BINS as f32;
            let hi_t = (bin + 1) as f32 / AUDIO_BINS as f32;
            let lo_hz = (min_log + (max_log - min_log) * lo_t).exp();
            let hi_hz = (min_log + (max_log - min_log) * hi_t).exp();
            let lo = hz_to_index(lo_hz).max(1);
            let hi = hz_to_index(hi_hz).max(lo + 1).min(FFT_SIZE / 2);

            // Peak magnitude in the band → dB → normalized 0..1.
            let mut peak = 0.0_f32;
            for i in lo..hi {
                peak = peak.max(self.scratch[i].norm());
            }
            let mag = peak / (FFT_SIZE as f32 * 0.5);
            let db = 20.0 * mag.max(1e-6).log10();
            let raw = ((db + 78.0) / 72.0).clamp(0.0, 1.0);

            // Fast attack, slow release — punchy but not jittery.
            let prev = self.last[bin];
            *out = if raw > prev {
                prev * 0.30 + raw * 0.70
            } else {
                prev * 0.82 + raw * 0.18
            };
        }
        self.last = bins;
        bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure tone should light the spectrum bin covering its frequency more
    /// than the band edges — sanity-checks the windowing + log binning.
    #[test]
    fn pure_tone_peaks_in_its_band() {
        let rate = 48_000;
        let mut proc = SpectrumProcessor::new(rate);
        let freq = 1000.0_f32;
        // Enough samples for several hops so the smoother settles.
        let n = FFT_SIZE * 8;
        let tone: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / rate as f32).sin())
            .collect();
        let bins = proc.push(&tone).expect("a full window of samples");

        // Which log-spaced bin should 1 kHz fall in?
        let (min_log, max_log) = (MIN_HZ.ln(), MAX_HZ.min(rate as f32 * 0.5).ln());
        let t = (freq.ln() - min_log) / (max_log - min_log);
        let expected = (t * AUDIO_BINS as f32) as usize;

        let loudest = bins
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Allow ±1 bin for rounding at the band edges.
        assert!(
            (loudest as i32 - expected as i32).abs() <= 1,
            "1 kHz landed in bin {loudest}, expected ~{expected}"
        );
    }

    #[test]
    fn silence_is_zero() {
        let mut proc = SpectrumProcessor::new(48_000);
        let bins = proc.push(&vec![0.0; FFT_SIZE * 2]).expect("a full window");
        assert!(bins.iter().all(|&b| b == 0.0), "silence should be all zero");
        let u = pack(&bins);
        assert_eq!(u.level, 0.0);
    }

    #[test]
    fn pack_places_bins_and_bands() {
        let mut bins = [0.0_f32; AUDIO_BINS];
        bins[0] = 1.0; // a bass bin
        bins[AUDIO_BINS - 1] = 0.5; // a treble bin
        let u = pack(&bins);
        assert_eq!(u.bins[0][0], 1.0);
        assert_eq!(u.bins[(AUDIO_BINS - 1) / 4][(AUDIO_BINS - 1) % 4], 0.5);
        assert!(u.bass > 0.0);
        assert!(u.treble > 0.0);
    }
}
