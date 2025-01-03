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

@_exported import _Differentiation
#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
@_exported import Numerics
#endif

#if !TENSORFLOW_USE_STANDARD_TOOLCHAIN
// MARK: - Array extensions

extension Array: ElementaryFunctions where Element: ElementaryFunctions {
  /// The square root of `x`.
  public static func sqrt(_ x: Self) -> Self { x.map(Element.sqrt) }
  /// The cosine of `x`, interpreted as an angle in radians.
  public static func cos(_ x: Self) -> Self { x.map(Element.cos) }
  /// The sine of `x`, interpreted as an angle in radians.
  public static func sin(_ x: Self) -> Self { x.map(Element.sin) }
  /// The tangent of `x`, interpreted as an angle in radians.
  public static func tan(_ x: Self) -> Self { x.map(Element.tan) }
  /// The inverse cosine of `x` in radians.
  public static func acos(_ x: Self) -> Self { x.map(Element.acos) }
  /// The inverse sine of `x` in radians.
  public static func asin(_ x: Self) -> Self { x.map(Element.asin) }
  /// The inverse tangent of `x` in radians.
  public static func atan(_ x: Self) -> Self { x.map(Element.atan) }
  /// The hyperbolic cosine of `x`.
  public static func cosh(_ x: Self) -> Self { x.map(Element.cosh) }
  /// The hyperbolic sine of `x`.
  public static func sinh(_ x: Self) -> Self { x.map(Element.sinh) }
  /// The hyperbolic tangent of `x`.
  public static func tanh(_ x: Self) -> Self { x.map(Element.tanh) }
  /// The inverse hyperbolic cosine of `x`.
  public static func acosh(_ x: Self) -> Self { x.map(Element.acosh) }
  /// The inverse hyperbolic sine of `x`.
  public static func asinh(_ x: Self) -> Self { x.map(Element.asinh) }
  /// The inverse hyperbolic tangent of `x`.
  public static func atanh(_ x: Self) -> Self { x.map(Element.atanh) }
  /// The exponential function applied to `x`, or `e**x`.
  public static func exp(_ x: Self) -> Self { x.map(Element.exp) }
  /// Two raised to to power `x`.
  public static func exp2(_ x: Self) -> Self { x.map(Element.exp2) }
  /// Ten raised to to power `x`.
  public static func exp10(_ x: Self) -> Self { x.map(Element.exp10) }
  /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
  public static func expm1(_ x: Self) -> Self { x.map(Element.expm1) }
  /// The natural logarithm of `x`.
  public static func log(_ x: Self) -> Self { x.map(Element.log) }
  /// The base-two logarithm of `x`.
  public static func log2(_ x: Self) -> Self { x.map(Element.log2) }
  /// The base-ten logarithm of `x`.
  public static func log10(_ x: Self) -> Self { x.map(Element.log10) }
  /// `log(1 + x)` evaluated so as to preserve accuracy close to zero.
  public static func log1p(_ x: Self) -> Self { x.map(Element.log1p) }

  /// `exp(y log(x))` computed without loss of intermediate precision.
  public static func pow(_ x: Self, _ y: Self) -> Self {
    precondition(x.count == y.count)
    return zip(x, y).map(Element.pow)
  }

  /// `x` raised to the `n`th power.
  public static func pow(_ x: Self, _ n: Int) -> Self { x.map { Element.pow($0, n) } }

  /// The `n`th root of `x`.
  public static func root(_ x: Self, _ n: Int) -> Self { x.map { Element.root($0, n) } }
}
#endif

// MARK: - Array derivative extensions

extension Array.DifferentiableView: ElementaryFunctions
where Element: Differentiable & ElementaryFunctions {
  /// The square root of `x`.
  public static func sqrt(_ x: Self) -> Self { .init(x.map(Element.sqrt)) }
  /// The cosine of `x`, interpreted as an angle in radians.
  public static func cos(_ x: Self) -> Self { .init(x.map(Element.cos)) }
  /// The sine of `x`, interpreted as an angle in radians.
  public static func sin(_ x: Self) -> Self { .init(x.map(Element.sin)) }
  /// The tangent of `x`, interpreted as an angle in radians.
  public static func tan(_ x: Self) -> Self { .init(x.map(Element.tan)) }
  /// The inverse cosine of `x` in radians.
  public static func acos(_ x: Self) -> Self { .init(x.map(Element.acos)) }
  /// The inverse sine of `x` in radians.
  public static func asin(_ x: Self) -> Self { .init(x.map(Element.asin)) }
  /// The inverse tangent of `x` in radians.
  public static func atan(_ x: Self) -> Self { .init(x.map(Element.atan)) }
  /// The hyperbolic cosine of `x`.
  public static func cosh(_ x: Self) -> Self { .init(x.map(Element.cosh)) }
  /// The hyperbolic sine of `x`.
  public static func sinh(_ x: Self) -> Self { .init(x.map(Element.sinh)) }
  /// The hyperbolic tangent of `x`.
  public static func tanh(_ x: Self) -> Self { .init(x.map(Element.tanh)) }
  /// The inverse hyperbolic cosine of `x`.
  public static func acosh(_ x: Self) -> Self { .init(x.map(Element.acosh)) }
  /// The inverse hyperbolic sine of `x`.
  public static func asinh(_ x: Self) -> Self { .init(x.map(Element.asinh)) }
  /// The inverse hyperbolic tangent of `x`.
  public static func atanh(_ x: Self) -> Self { .init(x.map(Element.atanh)) }
  /// The exponential function applied to `x`, or `e**x`.
  public static func exp(_ x: Self) -> Self { .init(x.map(Element.exp)) }

#if !TENSORFLOW_USE_STANDARD_TOOLCHAIN
  /// Two raised to to power `x`.
  public static func exp2(_ x: Self) -> Self { .init(Array.exp2(x.base)) }

  /// Ten raised to to power `x`.
  public static func exp10(_ x: Self) -> Self { .init(Array.exp10(x.base)) }

  /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
  public static func expm1(_ x: Self) -> Self { .init(Array.expm1(x.base)) }
#else
  /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
  public static func expMinusOne(_ x: Self) -> Self {
    .init(x.map(Element.expMinusOne))
  }
#endif

  /// The natural logarithm of `x`.
  /// (NOTE: This line looks suspiciously like it uses `Element.exp($0)` instead of `Element.log($0)`—fix if needed!)
  public static func log(_ x: Self) -> Self { .init(x.map { Element.log($0) }) }

#if !TENSORFLOW_USE_STANDARD_TOOLCHAIN
  /// The base-two logarithm of `x`.
  public static func log2(_ x: Self) -> Self { .init(Array.log2(x.base)) }

  /// The base-ten logarithm of `x`.
  public static func log10(_ x: Self) -> Self { .init(Array.log10(x.base)) }

  /// `log(1 + x)` evaluated so as to preserve accuracy close to zero.
  public static func log1p(_ x: Self) -> Self {
    .init(Array.log1p(x.base))
  }
#else
  /// The natural logarithm of `x + 1` to preserve accuracy close to zero.
  public static func log(onePlus x: Self) -> Self {
    .init(x.map { Element.log(onePlus: $0) })
  }
#endif

  /// `exp(y log(x))` computed without loss of intermediate precision.
  public static func pow(_ x: Self, _ y: Self) -> Self {
    .init(zip(x, y).map(Element.pow))
  }

  /// `x` raised to the `n`th power.
  public static func pow(_ x: Self, _ n: Int) -> Self {
    .init(x.map { Element.pow($0, n) })
  }

  /// The `n`th root of `x`.
  public static func root(_ x: Self, _ n: Int) -> Self {
    .init(x.map { Element.root($0, n) })
  }
}

// MARK: - Collection conformances

extension Array.DifferentiableView:
  BidirectionalCollection,
  Collection,
  MutableCollection,
  RandomAccessCollection,
  RangeReplaceableCollection,
  Sequence
where Element: Differentiable
{
  public typealias Element = Array<Element>.Element
  public typealias Index = Array<Element>.Index
  public typealias Indices = Array<Element>.Indices
  public typealias SubSequence = Array<Element>.SubSequence

  @inlinable
  public var startIndex: Index { base.startIndex }

  @inlinable
  public var endIndex: Index { base.endIndex }

  // For BidirectionalCollection & RandomAccessCollection:
  @inlinable
  public func index(after i: Index) -> Index {
    base.index(after: i)
  }

  @inlinable
  public func index(before i: Index) -> Index {
    base.index(before: i)
  }

  @inlinable
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    base.index(i, offsetBy: distance)
  }

  @inlinable
  public func distance(from start: Index, to end: Index) -> Int {
    base.distance(from: start, to: end)
  }

  // Core MutableCollection subscript
  @inlinable
  public subscript(position: Index) -> Element {
    _read { yield base[position] }
    set { base[position] = newValue }
  }

  // Range subscript for SubSequence
  @inlinable
  public subscript(bounds: Range<Index>) -> SubSequence {
    get { base[bounds] }
    set { base[bounds] = newValue }
  }

  // RangeReplaceableCollection requirement
  @inlinable
  public mutating func replaceSubrange<C>(
    _ subrange: Range<Index>,
    with newElements: C
  ) where C: Collection, Element == C.Element {
    base.replaceSubrange(subrange, with: newElements)
  }

  // You already have an init that calls `self.init(.init())`.
  // For clarity, here is a simple empty initializer:
  @inlinable
  public init() {
    self.init([])
  }
}

// MARK: - VectorProtocol conformance

extension Array.DifferentiableView: VectorProtocol
where Element: Differentiable & VectorProtocol {
  public typealias VectorSpaceScalar = Element.VectorSpaceScalar

  public func adding(_ x: VectorSpaceScalar) -> Self {
    .init(map { $0.adding(x) })
  }

  public mutating func add(_ x: VectorSpaceScalar) {
    for i in indices {
      self[i].add(x)
    }
  }

  public func subtracting(_ x: VectorSpaceScalar) -> Self {
    .init(map { $0.subtracting(x) })
  }

  public mutating func subtract(_ x: VectorSpaceScalar) {
    for i in indices {
      self[i].subtract(x)
    }
  }

  public func scaled(by scale: VectorSpaceScalar) -> Self {
    .init(map { $0.scaled(by: scale) })
  }

  public mutating func scale(by scale: VectorSpaceScalar) {
    for i in indices {
      self[i].scale(by: scale)
    }
  }
}

// MARK: - PointwiseMultiplicative conformance

extension Array.DifferentiableView: PointwiseMultiplicative
where Element: Differentiable & PointwiseMultiplicative {
  // `one` doesn't make sense for an Array, so this will trap.
  public static var one: Self {
    fatalError("`one` is not array-representable.")
  }

  public var reciprocal: Self {
    .init(map { $0.reciprocal })
  }

  public static func .* (lhs: Self, rhs: Self) -> Self {
    precondition(lhs.count == rhs.count, "Count mismatch: \(lhs.count) vs. \(rhs.count)")
    return .init(zip(lhs, rhs).map { $0 .* $1 })
  }

  public static func .*= (lhs: inout Self, rhs: Self) {
    precondition(lhs.count == rhs.count, "Count mismatch: \(lhs.count) vs. \(rhs.count)")
    for (i, x) in zip(lhs.indices, rhs) {
      lhs[i] .*= x
    }
  }
}

// MARK: - Index helper

extension Collection {
  /// Returns the `n`th position in `self`, offset from `startIndex`.
  func index(atOffset n: Int) -> Index {
    index(startIndex, offsetBy: n)
  }
}
