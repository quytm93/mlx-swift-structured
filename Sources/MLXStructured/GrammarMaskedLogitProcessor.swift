//
//  GrammarMaskedLogitProcessor.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 14.09.2025.
//

import MLXLMCommon
import MLX

public final class GrammarMaskedLogitProcessor: LogitProcessor, @unchecked Sendable {

    public let grammarMatcher: GrammarMatcher

    public init(grammarMatcher: GrammarMatcher) {
        self.grammarMatcher = grammarMatcher
    }

    public func prompt(_ prompt: MLXArray) {
        grammarMatcher.reset()
    }

    public func process(logits: MLXArray) -> MLXArray {
        return logits + normalizedMask(for: logits)
    }

    public func didSample(token: MLXArray) {
        if !grammarMatcher.isTerminated() {
            grammarMatcher.advance(token: token)
        }
    }

    private func normalizedMask(for logits: MLXArray) -> MLXArray {
        let mask = grammarMatcher.nextTokenMask()
        guard let logitsWidth = logits.shape.last else {
            return mask
        }

        let maskWidth = mask.size
        if maskWidth == logitsWidth {
            return mask
        }

        if maskWidth < logitsWidth {
            let padding = full([logitsWidth - maskWidth], values: -Float.infinity)
            return concatenated([mask, padding])
        }

        return mask[0..<logitsWidth]
    }
}
