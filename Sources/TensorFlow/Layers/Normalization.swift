import _Differentiation

// MARK: - A helper for normalizing

@differentiable(reverse)
private func normalize<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  mean: Tensor<Scalar>,
  variance: Tensor<Scalar>,
  offset: Tensor<Scalar>,
  scale: Tensor<Scalar>,
  varianceEpsilon: Tensor<Scalar>
) -> Tensor<Scalar> {
  let inv = scale * rsqrt(variance + varianceEpsilon)
  return input * inv + (offset - mean * inv)
}

// MARK: - BatchNorm

/// A batch normalization layer.
///
/// Normalizes the activations of the previous layer at each batch, i.e. applies a transformation
/// that maintains the mean activation close to `0` and the activation standard deviation close to `1`.
///
/// Reference: [Batch Normalization](https://arxiv.org/abs/1502.03167).
@frozen
public struct BatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  // Satisfy the `Layer` protocol:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  /// The feature dimension.
  @noDerivative public let axis: Int
  /// The momentum for the running mean and running variance.
  @noDerivative public let momentum: Scalar
  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar
  /// The running mean.
  @noDerivative public var runningMean: Parameter<Scalar>
  /// The running variance.
  @noDerivative public var runningVariance: Parameter<Scalar>

  /// Creates a batch normalization layer.
  public init(
    axis: Int,
    momentum: Scalar,
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    epsilon: Scalar,
    runningMean: Tensor<Scalar>,
    runningVariance: Tensor<Scalar>
  ) {
    precondition(offset.rank == 1, "The offset must have rank 1.")
    precondition(scale.rank == 1, "The scale must have rank 1.")
    precondition(offset.shape == scale.shape, "The offset and scale must have same shape.")
    self.axis = axis
    self.momentum = momentum
    self.offset = offset
    self.scale = scale
    self.epsilon = epsilon
    self.runningMean = Parameter(runningMean)
    self.runningVariance = Parameter(runningVariance)
  }

  /// Creates a batch normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis that should be normalized (typically the features axis).
  ///   - momentum: The momentum for the moving average.
  ///   - epsilon: The small scalar added to variance for numerical stability.
  public init(
    featureCount: Int,
    axis: Int = -1,
    momentum: Scalar = 0.99,
    epsilon: Scalar = 0.001
  ) {
    self.init(
      axis: axis,
      momentum: momentum,
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      epsilon: epsilon,
      runningMean: Tensor(0),
      runningVariance: Tensor(1)
    )
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Input) -> Output {
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "Number of features of the input and offset doesn't match."
    )

    // Reshape 'offset' and 'scale' for broadcasting if needed:
    var offset = self.offset
    var scale = self.scale
    if positiveAxis != input.rank - 1 {
      var broadcastShape = TensorShape(Array(repeating: 1, count: input.rank))
      broadcastShape[positiveAxis] = input.shape[positiveAxis]
      offset = offset.reshaped(to: broadcastShape)
      scale = scale.reshaped(to: broadcastShape)
    }

    // Depending on the learning phase, call doTraining vs doInference.
    switch Context.local.learningPhase {
    case .training:
      return doTraining(input, offset: offset, scale: scale, axis: positiveAxis)
    case .inference:
      return doInference(input, offset: offset, scale: scale)
    }
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }

  // ---------------------------------------
  // Implementation details:

  private func doTraining(
    _ input: Tensor<Scalar>,
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int
  ) -> Tensor<Scalar> {
    // 1) The axes to reduce are all except the batch and 'axis'.
    var normalizedAxes = Array(0..<input.rank)
    normalizedAxes.remove(at: axis)

    // 2) Compute mean & variance along those axes.
    let moments = input.moments(alongAxes: normalizedAxes)

    // 3) Compute the decay factor as a Tensor.
    let decayMomentum = Tensor(1 - momentum, on: input.device)

    // 4) Possibly handle reduced-precision scenarios:
    let isReducedPrecision = input.isReducedPrecision
    var momentsMean = moments.mean
    var momentsVariance = moments.variance
    if isReducedPrecision {
      momentsMean = momentsMean.toFullPrecision
      momentsVariance = momentsVariance.toFullPrecision
    }

    // 5) Update running stats in-place.
    runningMean.value += (momentsMean - runningMean.value) * decayMomentum
    runningVariance.value += (momentsVariance - runningVariance.value) * decayMomentum

    // 6) Epsilon: block gradient if needed:
    let eps = Tensor<Scalar>(epsilon, deviceAndPrecisionLike: input).stoppedGradient()

    return normalize(
      input,
      mean: moments.mean,
      variance: moments.variance,
      offset: offset,
      scale: scale,
      varianceEpsilon: eps
    )
  }

  private func doInference(
    _ input: Tensor<Scalar>,
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>
  ) -> Tensor<Scalar> {
    let isReducedPrecision = input.isReducedPrecision
    let runningVarianceValue = isReducedPrecision
      ? runningVariance.value.toReducedPrecision
      : runningVariance.value
    let runningMeanValue = isReducedPrecision
      ? runningMean.value.toReducedPrecision
      : runningMean.value
    let eps = Tensor<Scalar>(epsilon, deviceAndPrecisionLike: input).stoppedGradient()

    return normalize(
      input,
      mean: runningMeanValue,
      variance: runningVarianceValue,
      offset: offset,
      scale: scale,
      varianceEpsilon: eps
    )
  }
}

// MARK: - LayerNorm

/// A layer that applies layer normalization over a mini-batch of inputs.
///
/// Reference: [Layer Normalization](https://arxiv.org/abs/1607.06450).
@frozen
public struct LayerNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The axis for normalization.
  @noDerivative public let axis: Int
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar

  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int,
    epsilon: Scalar
  ) {
    precondition(offset.rank == 1, "Offset must have rank 1.")
    precondition(scale.rank == 1, "Scale must have rank 1.")
    precondition(offset.shape == scale.shape, "Offset & scale must have same shape.")
    self.offset = offset
    self.scale = scale
    self.axis = axis
    self.epsilon = epsilon
  }

  public init(
    featureCount: Int,
    axis: Int,
    epsilon: Scalar = 0.001
  ) {
    self.init(
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      axis: axis,
      epsilon: epsilon
    )
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Input) -> Output {
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "Number of features of input & offset doesn't match."
    )

    var broadcastShape = TensorShape(Array(repeating: 1, count: input.rank))
    broadcastShape[positiveAxis] = input.shape[positiveAxis]
    let offsetB = offset.reshaped(to: broadcastShape)
    let scaleB = scale.reshaped(to: broadcastShape)

    let moments = input.moments(alongAxes: positiveAxis)
    let eps = Tensor<Scalar>(epsilon, deviceAndPrecisionLike: input).stoppedGradient()
    let inv = rsqrt(moments.variance + eps) * scaleB
    return (input - moments.mean) * inv + offsetB
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - GroupNorm

/// A layer that applies group normalization over a mini-batch of inputs.
///
/// Reference: [Group Normalization](https://arxiv.org/abs/1803.08494).
@frozen
public struct GroupNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The number of groups.
  @noDerivative public let groupCount: Int
  /// The axis where features lie.
  @noDerivative public let axis: Int
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar

  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    groupCount: Int,
    axis: Int,
    epsilon: Scalar
  ) {
    precondition(axis != 0, "Axis cannot be the batch axis.")
    precondition(offset.rank == 1, "Offset must have rank 1.")
    precondition(
      offset.shape[0].isMultiple(of: groupCount),
      "Number of offset elements must be divisible by groupCount."
    )
    precondition(offset.shape == scale.shape, "Offset & scale must have same shape.")
    self.offset = offset
    self.scale = scale
    self.groupCount = groupCount
    self.axis = axis
    self.epsilon = epsilon
  }

  public init(
    featureCount: Int,
    groupCount: Int,
    axis: Int = -1,
    epsilon: Scalar = 1e-3
  ) {
    precondition(
      featureCount.isMultiple(of: groupCount),
      "Feature count must be divisible by groupCount."
    )
    self.init(
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      groupCount: groupCount,
      axis: axis,
      epsilon: epsilon
    )
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Input) -> Output {
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(positiveAxis != 0, "Axis cannot be batch axis.")
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "Feature count mismatch between input and offset."
    )

    // Reshape offset & scale for group broadcasting:
    var offsetB = offset
    var scaleB = scale
    var broadcastShape = TensorShape([Int](repeating: 1, count: input.rank + 1))
    broadcastShape[positiveAxis] = groupCount
    broadcastShape[positiveAxis + 1] = input.shape[positiveAxis] / groupCount
    offsetB = offsetB.reshaped(to: broadcastShape)
    scaleB = scaleB.reshaped(to: broadcastShape)

    // Reshape input so that its channel dimension is splitted: [groupCount, remainder].
    var groupShape = input.shape
    groupShape[positiveAxis] /= groupCount
    groupShape.insert(groupCount, at: positiveAxis)
    let grouped = input.reshaped(to: groupShape)

    // We'll compute mean/variance across all dims except the newly inserted dimension and batch.
    var normalizedAxes = Array(1..<grouped.rank)
    normalizedAxes.remove(at: positiveAxis - 1)
    let moments = grouped.moments(alongAxes: normalizedAxes)

    let eps = Tensor<Scalar>(epsilon, deviceAndPrecisionLike: input).stoppedGradient()
    let normalized = normalize(
      grouped,
      mean: moments.mean,
      variance: moments.variance,
      offset: offsetB,
      scale: scaleB,
      varianceEpsilon: eps
    )
    return normalized.reshaped(to: input.shape)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

// MARK: - InstanceNorm

/// A layer that applies instance normalization over a mini-batch of inputs.
///
/// Reference: [Instance Normalization](https://arxiv.org/abs/1607.08022).
@frozen
public struct InstanceNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  /// The underlying group normalization layer. InstanceNorm is a special case of GroupNorm
  /// with `groupCount` = number of features.
  var delegate: GroupNorm<Scalar>

  /// The offset (beta).
  public var offset: Tensor<Scalar> {
    _read { yield delegate.offset }
    _modify { yield &delegate.offset }
  }
  /// The scale (gamma).
  public var scale: Tensor<Scalar> {
    _read { yield delegate.scale }
    _modify { yield &delegate.scale }
  }
  public var axis: Int { delegate.axis }
  public var epsilon: Scalar { delegate.epsilon }

  /// Creates an instance normalization layer.
  ///
  /// - Parameters:
  ///   - offset: The initial offset value (beta).
  ///   - scale: The initial scale value (gamma).
  ///   - axis: The axis where features lie.
  ///   - epsilon: The variance epsilon value.
  /// - Precondition: The axis cannot be batch axis.
  /// - Precondition: The offset must have rank 1.
  /// - Precondition: offset.shape == scale.shape.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int,
    epsilon: Scalar
  ) {
    self.delegate = GroupNorm(
      offset: offset,
      scale: scale,
      groupCount: offset.shape[0],
      axis: axis,
      epsilon: epsilon
    )
  }

  /// Creates an instance normalization layer from featureCount.
  public init(
    featureCount: Int,
    axis: Int = -1,
    epsilon: Scalar = 1e-3
  ) {
    // groupCount = featureCount -> each channel is its own group => instance norm
    self.delegate = GroupNorm(
      featureCount: featureCount,
      groupCount: featureCount,
      axis: axis,
      epsilon: epsilon
    )
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Input) -> Output {
    delegate(input)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}
