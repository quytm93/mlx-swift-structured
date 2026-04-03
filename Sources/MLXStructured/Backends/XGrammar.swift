//
//  XGrammar.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 14.09.2025.
//

import Foundation
import CMLXStructured
import MLX

enum XGrammarError: Error {
    case emptyGrammar
    case invalidGrammar(String)
    case invalidVocab(String)
    case unknown(String)
}

enum CErrorHandler {

    private static let state = State()

    private static let installHandler: Void = {
        set_error_handler(errorHandlerClosure)
    }()

    private static let errorHandlerClosure: @convention(c) (UnsafePointer<CChar>?) -> Void = {
        state.lastErrorMessage = $0.map {
            String(cString: $0)
        }
    }

    static func initialize() {
        _ = installHandler
    }

    static func clearLastError() {
        state.lastErrorMessage = nil
    }

    static var lastErrorMessage: String {
        state.lastErrorMessage ?? "Unknown Error"
    }

    private final class State: @unchecked Sendable {

        let lock = NSLock()
        var _lastErrorMessage: String? = nil

        var lastErrorMessage: String? {
            get { lock.withLock { _lastErrorMessage } }
            set { lock.withLock { _lastErrorMessage = newValue } }
        }
    }
}

@inline(__always)
func withCErrorHandling<T>(_ body: () throws -> T) rethrows -> T {
    CErrorHandler.initialize()
    CErrorHandler.clearLastError()
    return try body()
}

final class XGrammar {

    private let vocabSize: Int
    private let bufferSize: Int
    private let bitmap: MLXArray
    private var bitmask: DLTensor
    private let grammarMatcher: UnsafeMutableRawPointer?

    init(
        vocab: [String],
        vocabType: Int32 = 0,
        stopTokenIds: [Int32] = [],
        grammar: Grammar
    ) throws {
        let vocab = vocab.map { strdup($0) }
        let tokenizerInfo = withCErrorHandling {
            vocab.map({ UnsafePointer($0) }).withUnsafeBufferPointer { vocabBuffer in
                stopTokenIds.withUnsafeBufferPointer { stopTokenIdsBuffer in
                    tokenizer_info_new(
                        vocabBuffer.baseAddress,
                        vocabBuffer.count,
                        vocabType,
                        stopTokenIdsBuffer.baseAddress,
                        stopTokenIdsBuffer.count
                    )
                }
            }
        }

        defer {
            tokenizer_info_free(tokenizerInfo)
            vocab.forEach {
                free($0)
            }
        }

        guard let tokenizerInfo else {
            throw XGrammarError.invalidVocab(CErrorHandler.lastErrorMessage)
        }

        let compiledGrammar = try withCErrorHandling {
            switch grammar {
            case _ where grammar.raw.isEmpty:
                throw XGrammarError.emptyGrammar
            case .ebnf(let ebnf):
                return ebnf.utf8CString.withUnsafeBufferPointer {
                    compile_ebnf_grammar(tokenizerInfo, $0.baseAddress, $0.count)
                }
            case .regex(let regex):
                return regex.utf8CString.withUnsafeBufferPointer {
                    compile_regex_grammar(tokenizerInfo, $0.baseAddress, $0.count)
                }
            case .schema(let schema, let options):
                return schema.utf8CString.withUnsafeBufferPointer { schemaBuffer in
                    let separators = options.separators
                    var compileOptions = json_schema_compile_options_t(
                        indent: Int32(options.indent ?? -1),
                        any_whitespace: options.anyWhitespace ? 1 : 0,
                        strict_mode: options.strictMode ? 1 : 0,
                        max_whitespace_cnt: Int32(options.maxWhitespaceCount ?? -1),
                        has_separators: 0,
                        separators: json_schema_separators_t(
                            comma_separator_utf8: nil,
                            comma_separator_len: 0,
                            colon_separator_utf8: nil,
                            colon_separator_len: 0
                        )
                    )

                    return separators?.comma.utf8CString.withUnsafeBufferPointer { commaBuffer in
                        separators?.colon.utf8CString.withUnsafeBufferPointer { colonBuffer in
                            compileOptions.has_separators = 1
                            compileOptions.separators = json_schema_separators_t(
                                comma_separator_utf8: commaBuffer.baseAddress,
                                comma_separator_len: commaBuffer.count - 1,
                                colon_separator_utf8: colonBuffer.baseAddress,
                                colon_separator_len: colonBuffer.count - 1
                            )
                            return compile_json_schema_grammar(
                                tokenizerInfo,
                                schemaBuffer.baseAddress,
                                schemaBuffer.count,
                                &compileOptions
                            )
                        }
                    }
                        ?? compile_json_schema_grammar(
                            tokenizerInfo,
                            schemaBuffer.baseAddress,
                            schemaBuffer.count,
                            &compileOptions
                        )
                }
            case .structural(let tag):
                return tag.utf8CString.withUnsafeBufferPointer {
                    compile_structural_tag(tokenizerInfo, $0.baseAddress, $0.count)
                }
            }
        }

        defer {
            compiled_grammar_free(compiledGrammar)
        }

        guard let compiledGrammar else {
            throw XGrammarError.invalidGrammar(CErrorHandler.lastErrorMessage)
        }

        var bitmap = [Float](repeating: 0, count: 256 * 8)
        for b in 0..<256 {
            for k in 0..<8 {
                bitmap[b * 8 + k] = ((b >> k) & 1) == 1 ? 0 : -Float.infinity
            }
        }

        guard let grammarMatcher = withCErrorHandling({ grammar_matcher_new(compiledGrammar) }) else {
            throw XGrammarError.unknown(CErrorHandler.lastErrorMessage)
        }

        self.vocabSize = vocab.count
        self.bufferSize = (vocab.count + 31) / 32
        self.bitmap = MLXArray(bitmap).reshaped([256, 8])
        self.bitmask = DLTensor.nextTokenBitmask(bufferSize: bufferSize)
        self.grammarMatcher = grammarMatcher
    }

    convenience init(tokenizerInfo: TokenizerInfo, grammar: Grammar) throws {
        try self.init(
            vocab: tokenizerInfo.vocab,
            vocabType: tokenizerInfo.vocabType,
            stopTokenIds: tokenizerInfo.stopTokenIds,
            grammar: grammar
        )
    }

    deinit {
        bitmask.data?.deallocate()
        bitmask.shape?.deallocate()
        bitmask.strides?.deallocate()
        grammar_matcher_free(grammarMatcher)
    }
}

extension XGrammar: GrammarMatcher {

    func nextTokenMask() -> MLXArray {
        guard
            withUnsafeMutablePointer(
                to: &bitmask,
                {
                    grammar_matcher_fill_next_token_bitmask(grammarMatcher, $0)
                }
            )
        else {
            return MLXArray.zeros([vocabSize])
        }

        let bytes = bufferSize &<< 2
        let bitmaskData = UnsafeRawBufferPointer(start: bitmask.data, count: bytes)
        let bitmask = MLXArray(bitmaskData, [bytes], type: Int8.self)
        let mask = bitmap[bitmask].reshaped([bytes * 8])[0..<vocabSize]
        return mask
    }

    func advance(token: MLXArray) {
        let tokenID = token.item(Int32.self)
        let accepted = grammar_matcher_accept_token(grammarMatcher, tokenID)
        if !accepted {
            reset()
        }
    }

    func reset() {
        grammar_matcher_reset(grammarMatcher)
    }

    func isTerminated() -> Bool {
        return grammar_matcher_is_terminated(grammarMatcher)
    }
}

private extension DLTensor {
    static func nextTokenBitmask(bufferSize: Int) -> DLTensor {
        let dataBytes = bufferSize * MemoryLayout<Int32>.stride
        let data = UnsafeMutableRawPointer.allocate(byteCount: dataBytes, alignment: 64)
        data.bindMemory(to: Int32.self, capacity: bufferSize).initialize(repeating: 0, count: bufferSize)

        let shape = UnsafeMutablePointer<Int64>.allocate(capacity: 1)
        shape.initialize(repeating: 0, count: 1)
        shape[0] = Int64(bufferSize)

        let device = DLDevice(deviceType: 1, deviceId: 0)
        let dtype = DLDataType(rawCode: 0, bits: 32, lanes: 1)

        return DLTensor(
            data: data,
            device: device,
            ndim: 1,
            dtype: dtype,
            shape: shape,
            strides: nil,
            byteOffset: 0
        )
    }
}
