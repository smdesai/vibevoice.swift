// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VibeVoice",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "VibeVoice", targets: ["VibeVoice"]),
        .executable(name: "vibevoiceCLI", targets: ["VibeVoiceCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(
            url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "VibeVoice",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "VibeVoiceCLI",
            dependencies: [
                "VibeVoice",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
    ]
)
