//
//  EmbeddingResponse.swift
//  database-server
//
//  Created by Ritesh Pakala on 10/26/25.
//  Based on: https://github.com/mzbac/swift-mlx-server

import Vapor

struct EmbeddingResponse: Content {
    var object: String = "list"
    // let data: [EmbeddingData]
    let model: String
    let usage: UsageData
    let document: Database.Document?
    let user: Database.User?
    let success: Bool
}

struct EmbeddingBatchResponse: Content {
    var object: String = "list"
    let model: String
    let usage: UsageData
    let success: Bool
    let user: Database.User?
}

struct EmbeddingData: Content {
    var object: String = "embedding"
    let embedding: EmbeddingOutput
    let index: Int
}

enum EmbeddingOutput: Codable {
    case floats([Float])
    case base64(String)

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let arr = try? container.decode([Float].self) {
            self = .floats(arr)
        } else if let str = try? container.decode(String.self) {
            self = .base64(str)
        } else {
            throw DecodingError.typeMismatch(
                EmbeddingOutput.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath, debugDescription: "Expected [Float] or String"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .floats(let arr):
            try container.encode(arr)
        case .base64(let str):
            try container.encode(str)
        }
    }
}

struct UsageData: Content {
    let prompt_tokens: Int
    let total_tokens: Int
    
    static var empty: UsageData {
        .init(prompt_tokens: 0, total_tokens: 0)
    }
}
