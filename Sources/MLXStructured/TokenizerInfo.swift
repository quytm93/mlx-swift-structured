//
//  TokenizerInfo.swift
//  mlx-swift-structured
//
//  Created by Ivan Petrukha on 03.04.2026.
//

import Foundation

struct TokenizerInfo: Sendable {
    let vocab: [String]
    let vocabType: Int32
    let stopTokenIds: [Int32]
}
