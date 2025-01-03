// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import _Differentiation

/// An embedding layer.
///
/// `Embedding` is effectively a lookup table that maps indices from a fixed vocabulary to 
/// fixed-size (dense) vector representations, e.g. `[[0], [3]] -> [[0.25, 0.1], [0.6, -0.2]]`.
@frozen
public struct Embedding<Scalar: TensorFlowFloatingPoint>: Module {
  // MARK: - Module requirements

  public typealias Input = Tensor<Int32>
  public typealias Output = Tensor<Scalar>

  // For `Module`, we must provide both `callAsFunction` and `forward` with 
  // `@differentiable(reverse, wrt: (self, input))`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Input) -> Output {
    // Each integer in `input` is replaced by its corresponding row in `embeddings`.
    embeddings.gathering(atIndices: input)
  }

  // MARK: - Stored properties

  /// A learnable lookup table that maps vocabulary indices to their dense vector representations.
  public var embeddings: Tensor<Scalar>

  // MARK: - Initializers

  /// Creates an `Embedding` layer with randomly initialized embeddings of shape
  /// `(vocabularySize, embeddingSize)` so that each vocabulary index is given a vector
  /// representation.
  ///
  /// - Parameters:
  ///   - vocabularySize: The number of distinct indices (words) in the vocabulary. This number
  ///     should be the largest integer index plus one.
  ///   - embeddingSize: The size of the embedding vector for each word.
  ///   - embeddingsInitializer: A closure that returns initial values for `embeddings`.
  public init(
    vocabularySize: Int,
    embeddingSize: Int,
    embeddingsInitializer: ParameterInitializer<Scalar> = { Tensor(randomUniform: $0) }
  ) {
    precondition(vocabularySize > 0, "The vocabulary size must be greater than 0.")
    precondition(embeddingSize > 0, "The embedding size must be greater than 0.")
    self.init(embeddings: embeddingsInitializer([vocabularySize, embeddingSize]))
  }

  /// Creates an `Embedding` layer from the provided embeddings. Useful for introducing
  /// pretrained embeddings into a model.
  ///
  /// - Parameter embeddings: The pretrained embeddings table of shape `[vocabularySize, embeddingSize]`.
  public init(embeddings: Tensor<Scalar>) {
    self.embeddings = embeddings
  }
}
