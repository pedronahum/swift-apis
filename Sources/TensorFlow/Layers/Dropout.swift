import _Differentiation
#if os(Windows)
  import func CRT.sqrt
#endif

// MARK: - Extension: droppingOut(probability:)

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Computes dropout for this tensor given a probability. Only used during training.
  @differentiable(reverse)
  fileprivate func droppingOut(probability: Double) -> Tensor {
    let noise = Tensor(randomUniform: shape, on: device)
    let keepMask = noise .>= Scalar(probability)
    let keepProbability = Scalar(1.0 - probability)
    return self * Tensor(keepMask) / Tensor(keepProbability, on: device)
  }
}

// MARK: - Dropout

/// A dropout layer. Randomly sets input elements to `0` with probability `p`,
/// scaling the rest by `1/(1-p)` so that the overall sum remains the same.
///
/// This acts as a regularization technique: it is active during training and does nothing during
/// inference.
@frozen
public struct Dropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  /// No parameters, so the move is a no-op.
  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let probability: Double

  /// Creates a dropout layer with the specified probability.
  ///
  /// - Parameter probability: Probability of dropping out each element. Must be in [0,1].
  public init(probability: Double) {
    precondition(
      0...1 ~= probability,
      "Probability must be in [0,1] but is \(probability)."
    )
    self.probability = probability
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
      return input.droppingOut(probability: probability)
    case .inference:
      // pass-through
      return input
    }
  }

  /// Satisfies `ParameterlessLayer` by delegating to `forward(_:)`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GaussianNoise

/// A layer that adds noise sampled from a normal distribution with mean = 0.
///
/// This is only active during training; in inference, it is a pass-through.
@frozen
public struct GaussianNoise<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let standardDeviation: Tensor<Scalar>

  /// Creates a Gaussian noise layer with the specified standard deviation.
  public init(standardDeviation: Scalar) {
    self.standardDeviation = Tensor<Scalar>(standardDeviation)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
      let noise = Tensor<Scalar>(
        randomNormal: input.shape,
        mean: Tensor<Scalar>(0),
        standardDeviation: self.standardDeviation)
      return input + noise
    case .inference:
      return input
    }
  }

  /// Satisfies `ParameterlessLayer` by delegating to `forward(_:)`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GaussianDropout

/// A layer that multiplies the input by noise sampled from a normal distribution with mean = 1.0.
///
/// This is only active during training; in inference, it is a pass-through.
/// Probability must be in [0,1].
///
/// `stddev = sqrt(probability / (1 - probability))`
@frozen
public struct GaussianDropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let probability: Scalar
  @noDerivative public let standardDeviation: Scalar

  public init(probability: Scalar) {
    precondition(
      0...1 ~= probability,
      "Probability must be in [0,1], but is \(probability)."
    )
    self.probability = probability
    // stddev = sqrt(probability / (1 - probability))
    self.standardDeviation = sqrt(probability / (1 - probability))
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
      let noise = Tensor<Scalar>(
        randomNormal: input.shape,
        mean: Tensor<Scalar>(1.0),
        standardDeviation: Tensor<Scalar>(standardDeviation))
      return input * noise
    case .inference:
      return input
    }
  }

  /// Satisfies `ParameterlessLayer`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - AlphaDropout

/// AlphaDropout is a dropout variant that keeps mean and variance of inputs
/// close to original values, preserving "self-normalizing" properties.
/// It's typically used with SELU activations. It randomly sets inputs to
/// negative saturation value with probability = `probability`.
@frozen
public struct AlphaDropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let probability: Double

  /// Creates an AlphaDropout layer with the specified probability.
  /// Probability must be in [0,1].
  public init(probability: Double) {
    precondition(
      0...1 ~= probability,
      "Probability must be in [0,1], but is \(probability)."
    )
    self.probability = probability
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
      let alpha = Scalar(1.6732632423543772848170429916717)
      let scale = Scalar(1.0507009873554804934193349852946)
      let alphaP = -alpha * scale
      // The probability for dropping out each element is `probability`.
      let uniform = Tensor<Scalar>(randomUniform: input.shape, on: input.device)
      let keepMask = uniform .>= Scalar(probability)

      // Next, we compute the "a" and "b" constants from the original paper:
      let a = Scalar(1.0) / sqrt((Scalar(1.0) - Scalar(probability)) + Scalar(probability) * alphaP * alphaP)
      let b = -a * alphaP * Scalar(probability)

      // Where mask is false, we set to alphaP; where mask is true, we keep input.
      var x = input * Tensor(keepMask)
      x = x + alphaP * (1 - Tensor(keepMask))

      return a * x + b
    case .inference:
      return input
    }
  }

  /// Satisfies `ParameterlessLayer` by delegating to `forward`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}
