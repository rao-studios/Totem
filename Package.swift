// swift-tools-version: 6.0

import PackageDescription

var packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/apple/swift-argument-parser.git", .upToNextMajor(from: "1.3.0")),
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
    .package(url: "https://github.com/apple/swift-crypto.git", from: "4.2.0"),
    .package(url: "https://github.com/grpc/grpc-swift.git", from: "2.0.0"),
    .package(url: "https://github.com/grpc/grpc-swift-nio-transport.git", from: "1.0.0"),
    .package(url: "https://github.com/grpc/grpc-swift-protobuf.git", from: "1.0.0"),
    .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.28.0"),
]

var targetDependencies: [Target.Dependency] = [
    .product(name: "ArgumentParser", package: "swift-argument-parser"),
    .product(name: "Hummingbird", package: "hummingbird"),
    .product(name: "Crypto", package: "swift-crypto"),
    .product(name: "GRPCCore", package: "grpc-swift"),
    .product(name: "GRPCNIOTransportHTTP2", package: "grpc-swift-nio-transport"),
    .product(name: "GRPCProtobuf", package: "grpc-swift-protobuf"),
    .product(name: "SwiftProtobuf", package: "swift-protobuf"),
]

#if os(macOS)
packageDependencies += [
    .package(url: "https://github.com/riteshpakala/mlx.embeddings.git", branch: "main"),
    .package(url: "https://github.com/riteshpakala/mlx-swift-lm", branch: "main"),
]
targetDependencies += [
    .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
    .product(name: "mlx_embeddings", package: "mlx.embeddings"),
]
#endif

let supportedPlatforms: [SupportedPlatform] = [.macOS(.v15)]

let package = Package(
  name: "totem",
  platforms: supportedPlatforms,
  dependencies: packageDependencies,
  targets: [
    .executableTarget(
      name: "totem",
      dependencies: targetDependencies,
      path: "Sources"
    ),
    .testTarget(
      name: "totem-tests",
      dependencies: [
        "totem",
        .product(name: "HummingbirdTesting", package: "hummingbird"),
      ],
      path: "Tests/totem-tests"
    )
  ]
)
