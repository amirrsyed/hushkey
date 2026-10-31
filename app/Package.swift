// swift-tools-version:5.9
import PackageDescription

// `swift build` compiles the app; `scripts/build_app.sh` wraps the binary into Hushkey.app
// with Info.plist + entitlements (macOS only grants camera access to a bundled, signed app).
// Prefer Xcode? `brew install xcodegen && xcodegen generate` uses project.yml instead.
let package = Package(
    name: "Hushkey",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "Hushkey",
            path: "Sources/Hushkey"
        ),
    ],
    swiftLanguageVersions: [.v5]
)
