# Maintainer: Christian Balcom <robot.inventor@gmail.com>

pkgname=prism-bg
pkgver=0.1.0
pkgrel=1
pkgdesc='Color-managed wallpaper client for Wayland — swaybg, in Rust, HDR-aware'
arch=('x86_64')
url='https://github.com/computer-whisperer/prism-bg'
license=('MIT OR Apache-2.0')
depends=(
    'dav1d'
    'gcc-libs'
    'glibc'
)
makedepends=('cargo' 'clang' 'pkgconf')
# Disable system LTO — Arch's default `-flto=auto` lands in CFLAGS and makes
# the vendored jpegxr C sources (compiled by its build.rs via the `cc` crate)
# emit LTO-IR objects, which rust-lld can't resolve at the final Rust link step.
options=('!lto')
source=("$pkgname-$pkgver.tar.gz::$url/archive/refs/tags/v$pkgver.tar.gz")
# Run `updpkgsums` once the v0.1.0 tag exists on GitHub.
sha256sums=('SKIP')

prepare() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    cargo fetch --locked --target "$(rustc -vV | sed -n 's/host: //p')"
}

build() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    export CARGO_TARGET_DIR=target
    cargo build --release --frozen
}

check() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    cargo test --release --frozen
}

package() {
    cd "$pkgname-$pkgver"
    install -Dm755 "target/release/prism-bg" "$pkgdir/usr/bin/prism-bg"
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
    install -Dm644 LICENSE-MIT "$pkgdir/usr/share/licenses/$pkgname/LICENSE-MIT"
    install -Dm644 LICENSE-APACHE "$pkgdir/usr/share/licenses/$pkgname/LICENSE-APACHE"
}
