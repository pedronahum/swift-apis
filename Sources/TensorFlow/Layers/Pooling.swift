import _Differentiation

// MARK: - MaxPool1D

/// A max pooling layer for temporal data.
@frozen
public struct MaxPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) { /* No parameters */ }

  @noDerivative public let poolSize: Int
  @noDerivative public let stride: Int
  @noDerivative public let padding: Padding

  public init(poolSize: Int, stride: Int, padding: Padding) {
    precondition(poolSize > 0, "The pooling window size must be > 0.")
    precondition(stride > 0, "The stride must be > 0.")
    self.poolSize = poolSize
    self.stride = stride
    self.padding = padding
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    // We do a 2D maxPool with shape expanded at axis=1, then squeeze it back.
    maxPool2D(
      input.expandingShape(at: 1),
      filterSize: (1, 1, poolSize, 1),
      strides: (1, 1, stride, 1),
      padding: padding
    ).squeezingShape(at: 1)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - MaxPool2D

/// A max pooling layer for spatial data.
@frozen
public struct MaxPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let poolSize: (Int, Int, Int, Int)
  @noDerivative public let strides: (Int, Int, Int, Int)
  @noDerivative public let padding: Padding

  public init(poolSize: (Int, Int, Int, Int), strides: (Int, Int, Int, Int), padding: Padding) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0,
      "Pooling window sizes must be > 0."
    )
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0,
      "Strides must be > 0."
    )
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    maxPool2D(input, filterSize: poolSize, strides: strides, padding: padding)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

extension MaxPool2D {
  public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, 1),
      strides: (1, strides.0, strides.1, 1),
      padding: padding
    )
  }
}

// MARK: - MaxPool3D

/// A max pooling layer for spatial or spatio-temporal data.
@frozen
public struct MaxPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let poolSize: (Int, Int, Int, Int, Int)
  @noDerivative public let strides: (Int, Int, Int, Int, Int)
  @noDerivative public let padding: Padding

  public init(
    poolSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
  ) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0 && poolSize.4 > 0,
      "Pooling window sizes must be > 0."
    )
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0 && strides.4 > 0,
      "Strides must be > 0."
    )
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    maxPool3D(input, filterSize: poolSize, strides: strides, padding: padding)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

extension MaxPool3D {
  public init(poolSize: (Int, Int, Int), strides: (Int, Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, poolSize.2, 1),
      strides: (1, strides.0, strides.1, strides.2, 1),
      padding: padding
    )
  }

  public init(poolSize: Int, stride: Int, padding: Padding = .valid) {
    self.init(
      poolSize: (poolSize, poolSize, poolSize),
      strides: (stride, stride, stride),
      padding: padding
    )
  }
}

// MARK: - AvgPool1D

/// An average pooling layer for temporal data.
@frozen
public struct AvgPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let poolSize: Int
  @noDerivative public let stride: Int
  @noDerivative public let padding: Padding

  public init(poolSize: Int, stride: Int, padding: Padding) {
    precondition(poolSize > 0, "poolSize must be > 0.")
    precondition(stride > 0, "stride must be > 0.")
    self.poolSize = poolSize
    self.stride = stride
    self.padding = padding
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    avgPool2D(
      input.expandingShape(at: 1),
      filterSize: (1, 1, poolSize, 1),
      strides: (1, 1, stride, 1),
      padding: padding
    ).squeezingShape(at: 1)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - AvgPool2D

/// An average pooling layer for spatial data.
@frozen
public struct AvgPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let poolSize: (Int, Int, Int, Int)
  @noDerivative public let strides: (Int, Int, Int, Int)
  @noDerivative public let padding: Padding

  public init(poolSize: (Int, Int, Int, Int), strides: (Int, Int, Int, Int), padding: Padding) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0,
      "poolSize must be > 0."
    )
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0,
      "strides must be > 0."
    )
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    avgPool2D(input, filterSize: poolSize, strides: strides, padding: padding)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

extension AvgPool2D {
  public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, 1),
      strides: (1, strides.0, strides.1, 1),
      padding: padding
    )
  }
}

// MARK: - AvgPool3D

/// An average pooling layer for spatial or spatio-temporal data.
@frozen
public struct AvgPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let poolSize: (Int, Int, Int, Int, Int)
  @noDerivative public let strides: (Int, Int, Int, Int, Int)
  @noDerivative public let padding: Padding

  public init(
    poolSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
  ) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0 && poolSize.4 > 0,
      "poolSize must be > 0."
    )
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0 && strides.4 > 0,
      "strides must be > 0."
    )
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    avgPool3D(input, filterSize: poolSize, strides: strides, padding: padding)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

extension AvgPool3D {
  public init(poolSize: (Int, Int, Int), strides: (Int, Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, poolSize.2, 1),
      strides: (1, strides.0, strides.1, strides.2, 1),
      padding: padding
    )
  }

  public init(poolSize: Int, strides: Int, padding: Padding = .valid) {
    self.init(
      poolSize: (poolSize, poolSize, poolSize),
      strides: (strides, strides, strides),
      padding: padding
    )
  }
}

// MARK: - GlobalAvgPool1D

/// A global average pooling layer for temporal data.
@frozen
public struct GlobalAvgPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  public init() {}

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 3, "Input must be rank-3 [batch, time, channels].")
    return input.mean(squeezingAxes: 1)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GlobalAvgPool2D

/// A global average pooling layer for spatial data.
@frozen
public struct GlobalAvgPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  public init() {}

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 4, "Input must be rank-4 [batch, height, width, channels].")
    return input.mean(squeezingAxes: [1, 2])
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GlobalAvgPool3D

/// A global average pooling layer for spatial and spatio-temporal data.
@frozen
public struct GlobalAvgPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  public init() {}

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 5, "Input must be rank-5 [batch, depth, height, width, channels].")
    return input.mean(squeezingAxes: [1, 2, 3])
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GlobalMaxPool1D

/// A global max pooling layer for temporal data.
@frozen
public struct GlobalMaxPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  public init() {}

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 3, "Input must be rank-3 [batch, time, channels].")
    return input.max(squeezingAxes: 1)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GlobalMaxPool2D

/// A global max pooling layer for spatial data.
@frozen
public struct GlobalMaxPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  public init() {}

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 4, "Input must be rank-4 [batch, height, width, channels].")
    return input.max(squeezingAxes: [1, 2])
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GlobalMaxPool3D

/// A global max pooling layer for spatial and spatio-temporal data.
@frozen
public struct GlobalMaxPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  public init() {}

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 5, "Input must be rank-5 [batch, depth, height, width, channels].")
    return input.max(squeezingAxes: [1, 2, 3])
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - FractionalMaxPool2D

/// A fractional max pooling layer for spatial data.
/// Note: `FractionalMaxPool` does not have an XLA implementation, and thus may have performance implications.
@frozen
public struct FractionalMaxPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  /// Pooling ratios for each dimension of input shape [batch, height, width, channels].
  /// Note: pooling in only height and width is supported.
  @noDerivative public let poolingRatio: (Double, Double, Double, Double)
  @noDerivative public let pseudoRandom: Bool
  @noDerivative public let overlapping: Bool
  @noDerivative public let deterministic: Bool
  @noDerivative public let seed: Int64
  @noDerivative public let seed2: Int64

  public init(
    poolingRatio: (Double, Double, Double, Double),
    pseudoRandom: Bool = false,
    overlapping: Bool = false,
    deterministic: Bool = false,
    seed: Int64 = 0,
    seed2: Int64 = 0
  ) {
    precondition(
      poolingRatio.0 == 1.0 && poolingRatio.3 == 1.0,
      "Pooling on batch and channels dims not supported."
    )
    precondition(
      poolingRatio.1 >= 1.0 && poolingRatio.2 >= 1.0,
      "Pooling ratio for height & width must be >= 1.0"
    )
    self.poolingRatio = poolingRatio
    self.pseudoRandom = pseudoRandom
    self.overlapping = overlapping
    self.deterministic = deterministic
    self.seed = seed
    self.seed2 = seed2
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    fractionalMaxPool2D(
      input,
      poolingRatio: poolingRatio,
      pseudoRandom: pseudoRandom,
      overlapping: overlapping,
      deterministic: deterministic,
      seed: seed,
      seed2: seed2
    )
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

extension FractionalMaxPool2D {
  public init(
    poolingRatio: (Double, Double),
    pseudoRandom: Bool = false,
    overlapping: Bool = false,
    deterministic: Bool = false,
    seed: Int64 = 0,
    seed2: Int64 = 0
  ) {
    self.init(
      poolingRatio: (1.0, poolingRatio.0, poolingRatio.1, 1.0),
      pseudoRandom: pseudoRandom,
      overlapping: overlapping,
      deterministic: deterministic,
      seed: seed,
      seed2: seed2
    )
  }
}
