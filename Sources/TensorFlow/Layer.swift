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



//
//  Layer.swift
//  Example
//
//  This file defines `Module`, `Layer`, `ParameterlessLayer`, and related extensions.
//  It uses `@differentiable(reverse, wrt: (self, input))` to match Swift's requirement
//  when the protocol is declared in a separate file from the conforming types.
//

import _Differentiation
import Foundation
#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
import Numerics
#endif

/// A high-level “module” that maps `Input` to `Output`.
///
/// - `TangentVector` must be a type that conforms to:
///   - `VectorProtocol` (so it can scale/add) 
///   - `ElementaryFunctions`, `PointwiseMultiplicative`, 
///   - `KeyPathIterable` (to iterate over parameters, if any).
public protocol Module: EuclideanDifferentiable, KeyPathIterable
where
  TangentVector: VectorProtocol & ElementaryFunctions & PointwiseMultiplicative & KeyPathIterable
{
  /// The input type of this module.
  associatedtype Input

  /// The output type of this module, which must be `Differentiable`.
  associatedtype Output: Differentiable

  /// Returns the output obtained from applying the module to the given input.
  ///
  /// - Parameter input: The input to this module.
  /// - Returns: The module’s output.
  ///
  /// By specifying `@differentiable(reverse, wrt: (self, input))`, we guarantee that
  /// differentiation can happen with respect to both the module parameters (`self`) and
  /// the input.
  @differentiable(reverse, wrt: (self, input))
  func callAsFunction(_ input: Input) -> Output

  /// A secondary function that is also differentiable w.r.t. (self, input),
  /// typically an alternative or alias for `callAsFunction(_:)`.
  @differentiable(reverse, wrt: (self, input))
  func forward(_ input: Input) -> Output
}

// MARK: - Default Implementations for Module

extension Module {
  /// By default, `forward(_:)` just calls `callAsFunction(_:)`.
  /// We use the same `@differentiable` specification so it
  /// remains consistent with the protocol requirement.
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Input) -> Output {
    callAsFunction(input)
  }
}

// MARK: - Additional Module Extensions

extension Module where Input: TensorProtocol, Output: DifferentiableTensorProtocol {
  
  /*
  /// A convenience wrapper that calls `forward(_:)` and then
  /// optionally applies an “annotation” step for debugging/XLA.
  ///
  /// We again specify `@differentiable(reverse, wrt: (self, input))` so it can
  /// differentiate with respect to both module parameters and input.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    let activation = forward(input)
    return annotated(activation)
  }
  */

  /// Annotates `output`—helpful for XLA or debugging.
  @differentiable(reverse, wrt: (self, output))
  public func annotated(_ output: Output) -> Output {
    // You can store an annotation on `output` if desired.
    // For example, it might store a “type=LayerName” string.
    output.annotate("type=\(Self.self)")
  }

  /// A user-friendly “summary” method that shows details
  /// of the layer’s output shape, type, attributes, etc.
  public func summary(input: Input) -> String {
    let output = self.callAsFunction(input)
    return formatAnnotations(from: output)
  }

  /// Formats the annotation string from a `DifferentiableTensorProtocol`.
  ///
  /// This is purely for debug/visualization purposes.
  private func formatAnnotations(from tensor: Output) -> String {
    let rawAnnotations = tensor.annotations
    if rawAnnotations == Device.defaultTFEager.annotationsAvailable {
      // If no special annotations, just return them.
      return rawAnnotations
    }

    // Example: parse out lines matching `shape=... type=...`
    let lines = rawAnnotations.components(separatedBy: "\n")
    if lines.count < 3 {
      return ""
    }

    let pattern = "\\s*shape=(.+)\\s+type=([^\\s]+)(\\s+.+=.+)?$"
    let regex = try! NSRegularExpression(pattern: pattern)
    let relevant = lines.filter { $0.contains("shape=") }

    let contents = relevant.map { line -> String in
      let range = NSRange(line.startIndex..., in: line)
      guard let match = regex.firstMatch(in: line, range: range) else { return line }

      // Extract the “type=”, “shape=”, and “attributes=…”
      var content = ""
      if let typeRange = Range(match.range(at: 2), in: line) {
        content += line[typeRange]
      }
      content += "\t\t\t"
      if let shapeRange = Range(match.range(at: 1), in: line) {
        content += line[shapeRange]
      }
      content += "\t\t"
      if let attrRange = Range(match.range(at: 3), in: line) {
        content += line[attrRange]
      }
      return content
    }

    return """
    Layer                           Output Shape         Attributes
    =============================== ==================== ======================
    \(contents.joined(separator: "\n"))
    """
  }
}

// MARK: - The `Layer` protocol

/// A “neural network layer” that refines `Module` by requiring:
/// - `Input` is `Differentiable`.
/// - The same `callAsFunction(_:)` signature, but typically focusing on NNs.
public protocol Layer: Module where Input: Differentiable {
  /// We explicitly require `@differentiable(reverse, wrt: (self, input))` so that
  /// the layer can be differentiated w.r.t. its parameters (`self`) and the input.
  @differentiable(reverse, wrt: (self, input))
  func callAsFunction(_ input: Input) -> Output

  
  @differentiable(reverse, wrt: (self, input))
  func forward(_ input: Input) -> Output
}

extension Layer {
  public typealias Backpropagator = (
    _ direction: Output.TangentVector
  ) -> (
    layerGradient: TangentVector,
    inputGradient: Input.TangentVector
  )

  public func appliedForBackpropagation(to input: Input)
    -> (output: Output, backpropagator: Backpropagator)
  {
    #if TENSORFLOW_USE_STANDARD_TOOLCHAIN
      let (out, pullback) = _Differentiation.valueWithPullback(at: self, input) {
        layer, input in layer(input)
      }
    #else
      let (out, pullback) = Swift.valueWithPullback(at: self, input) {
        layer, input in layer(input)
      }
    #endif

    let backprop: Backpropagator = { direction in
      pullback(direction)
    }
    return (out, backprop)
  }
}


/// A parameter‐free neural network layer.
///
/// - `TangentVector == EmptyTangentVector`, meaning no trainable parameters.
public protocol ParameterlessLayer: Layer where TangentVector == EmptyTangentVector {
  /// The `callAsFunction(_:)` method must be differentiable w.r.t. (self, input),
  /// matching the same signature as in `Layer`, just acknowledging no parameters.
  @differentiable(reverse, wrt: (self, input))
  func callAsFunction(_ input: Input) -> Output
}

// MARK: - Default Implementations

extension ParameterlessLayer {
  /// Because there are no parameters, we define a no‐op `move(along:)`.
  public mutating func move(along direction: EmptyTangentVector) {}

  /// No parameter vectors to return.
  public var differentiableVectorView: EmptyTangentVector { EmptyTangentVector() }
}

// MARK: - Utility: EmptyTangentVector

/// A “no‐parameter” tangent vector, used by parameter‐free layers.
public struct EmptyTangentVector: 
  EuclideanDifferentiable,
  VectorProtocol,
  ElementaryFunctions,
  PointwiseMultiplicative,
  KeyPathIterable 
{
  public typealias VectorSpaceScalar = Float
  public typealias TangentVector = Self

  public init() {}

  public func adding(_ x: Float) -> Self { self }
  public mutating func add(_ x: Float) {}
  public func subtracting(_ x: Float) -> Self { self }
  public mutating func subtract(_ x: Float) {}
  public func scaled(by scalar: Float) -> Self { self }
  public mutating func scale(by scalar: Float) {}

  // PointwiseMultiplicative
  public static var one: Self { Self() }
  public var reciprocal: Self { Self() }
  public static func .* (lhs: Self, rhs: Self) -> Self { Self() }
  public static func .*= (lhs: inout Self, rhs: Self) {}
}

/// A “Parameter” references a mutable `Tensor<Scalar>`.
/// Often used for trainable parameters in a layer.
public final class Parameter<Scalar: TensorFlowScalar> {
  public var value: Tensor<Scalar>
  public init(_ value: Tensor<Scalar>) {
    self.value = value
  }
}
