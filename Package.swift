// swift-tools-version:5.3

import PackageDescription

let package = Package(
        name: "gonzales",
        platforms: [.macOS(.v10_15)],
        dependencies: [
            .package(name: "PNG", url: "https://github.com/kelvin13/png.git", from: "3.0.2"),
        ],
        targets: [
                .target(name: "gonzales",
                        dependencies: ["PNG", "exr", "ptex"],
                        linkerSettings: [.unsafeFlags(["-L/usr/local/lib", "-lPtex"])]
                ),
                .target(name: "exr",
                        dependencies: ["openexr"],
                        cxxSettings: [.unsafeFlags(["-I/usr/local/include/OpenEXR"])]),
                .target(name: "ptex",
                        dependencies: [],
                        cxxSettings: [.unsafeFlags(["-I/usr/local/include"])]),
                .systemLibrary(name: "openexr", pkgConfig: "OpenEXR"),
        ],
        cxxLanguageStandard: .cxx11
)
