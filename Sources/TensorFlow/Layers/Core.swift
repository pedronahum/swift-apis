
import _Differentiation

// MARK: - Flatten

/// A flatten layer.
///
/// A flatten layer flattens the input when applied, without affecting the batch size.
@frozen
public struct Flatten<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  // Conform to `ParameterlessLayer` => also `Layer`.
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public init() {}

  // No parameters => no-op move:
  public mutating func move(by offset: EmptyTangentVector) {}

  /// The main logic: flatten batch dimension but combine all others.
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let batchSize = input.shape[0]
    let remaining = input.shape[1..<input.rank].contiguousSize
    return input.reshaped(to: [batchSize, remaining])
  }

  /// Satisfies `ParameterlessLayer` by delegating to `forward(_:)`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - Reshape

/// A reshape layer, which reshapes inputs to a specified target shape.
@frozen
public struct Reshape<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  // No parameters => no-op move:
  public mutating func move(by offset: EmptyTangentVector) {}

  /// The target shape (as a 1-D tensor of Int32).
  @noDerivative public var shape: Tensor<Int32>

  /// TF-331 workaround: to keep Swift from optimizing out.
  @usableFromInline
  internal var _nontrivial = Tensor<Float>(0)

  /// Creates a reshape layer from an Int32 shape tensor.
  public init(shape: Tensor<Int32>) {
    self.shape = shape
  }

  /// Creates a reshape layer from a `TensorShape`.
  public init(_ shape: TensorShape) {
    self.init(shape: Tensor(shape.dimensions.map(Int32.init)))
  }

  /// The main logic: reshape input to `shape`.
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    input.reshaped(toShape: shape)
  }

  /// Satisfies `ParameterlessLayer`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - Function

/// A layer that encloses a custom differentiable function `Body: (Input) -> Output`.
///
/// This is parameterless, so typically used for functional transformations or debugging.
public struct Function<Input: Differentiable, Output: Differentiable>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector
  public typealias Body = @differentiable(reverse) (Input) -> Output

  // Conformance to `Layer` requires we define `Input`/`Output` again:
  // But we already have them as the generics, so just re-alias them:
  public typealias Input = Input
  public typealias Output = Output

  // No parameters => no-op move:
  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let body: Body

  public init(_ body: @escaping Body) {
    self.body = body
  }

  /// By default, we can directly implement `callAsFunction(_:)`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    body(input)
  }
}
