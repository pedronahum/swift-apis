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

/// Returns normalized `input`.
///
/// - Parameters:
///   - input: The tensor to be normalized.
///   - mean: The mean tensor.
///   - variance: The variance tensor.
///   - offset: The tensor to be added to normalized tensor.
///   - scale: The tensor to be applied to normalized tensor.
///   - varianceEpsilon: The small number to avoid dividing by 0.
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

/// A batch normalization layer.
///
/// Normalizes the activations of the previous layer at each batch, i.e. applies a transformation
/// that maintains the mean activation close to `0` and the activation standard deviation close to
/// `1`.
///
/// Reference: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
/// Covariate Shift](https://arxiv.org/abs/1502.03167).
@frozen
public struct BatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
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
  ///
  /// - Parameters:
  ///   - axis: The axis that should not be normalized (typically the feature axis).
  ///   - momentum: The momentum for the moving average.
  ///   - offset: The offset to be added to the normalized tensor.
  ///   - scale: The scale to multiply the normalized tensor by.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
  ///   - runningMean: The running mean.
  ///   - runningVariance: The running variance.
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
    precondition(
      offset.shape == scale.shape,
      "The offset and the scale must have same shape.")
    self.axis = axis
    self.momentum = momentum
    self.offset = offset
    self.scale = scale
    self.epsilon = epsilon
    self.runningMean = Parameter(runningMean)
    self.runningVariance = Parameter(runningVariance)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable(reverse)
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "The number of features of the input and the offset doesn't match.")
    var offset = self.offset
    var scale = self.scale
    if positiveAxis != input.rank - 1 {
      var broadcastShape = TensorShape([Int](repeating: 1, count: input.rank))
      broadcastShape[positiveAxis] = input.shape[positiveAxis]
      offset = offset.reshaped(to: broadcastShape)
      scale = scale.reshaped(to: broadcastShape)
    }
    switch Context.local.learningPhase {
    case .training:
      return doTraining(input, offset: offset, scale: scale, axis: positiveAxis)
    case .inference:
      return doInference(input, offset: offset, scale: scale)
    }
  }

  private func doTraining(
    _ input: Tensor<Scalar>, offset: Tensor<Scalar>, scale: Tensor<Scalar>, axis: Int
  ) -> Tensor<Scalar> {
    // 1) Figure out the axes to reduce.
    var normalizedAxes = Array(0..<input.rank)
    normalizedAxes.remove(at: axis)

    // 2) Compute moments along these axes.
    let moments = input.moments(alongAxes: normalizedAxes)

    // 3) Decay factor as a Tensor. 'momentum' presumably a stored property.
    let decayMomentum = Tensor(1 - momentum, on: input.device)

    // 4) 'isReducedPrecision' is just a Bool property; no gradient needed.
    let isReducedPrecision = input.isReducedPrecision

    // 5) Possibly cast moments to full precision if needed.
    var momentsMean = moments.mean
    var momentsVariance = moments.variance
    if isReducedPrecision {
      momentsMean = momentsMean.toFullPrecision
      momentsVariance = momentsVariance.toFullPrecision
    }

    // 6) Update running stats (runningMean, runningVariance presumably some references).
    runningMean.value += (momentsMean - runningMean.value) * decayMomentum
    runningVariance.value += (momentsVariance - runningVariance.value) * decayMomentum

    // 7) 'eps' is a small constant, but we need to block gradient to avoid
    //    X10 scalarization issues. So we call 'stopGradient'.
    let eps = Tensor<Scalar>(epsilon, deviceAndPrecisionLike: input).stoppedGradient()
    

    // 8) Finally, normalize the input.
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
    // We can read `isReducedPrecision` directly from the tensor without `withoutDerivative`.
    let isReducedPrecision = input.isReducedPrecision
    let runningVarianceValue = isReducedPrecision
      ? runningVariance.value.toReducedPrecision
      : runningVariance.value
    let runningMeanValue = isReducedPrecision
      ? runningMean.value.toReducedPrecision
      : runningMean.value

    // If you want to block the gradient for `eps` to avoid X10 scalarization issues,
    // call a custom `stopGradient` function (defined in the same extension or elsewhere).
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


  /// Creates a batch normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis that should be normalized (typically the features axis).
  ///   - momentum: The momentum for the moving average.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
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
      runningVariance: Tensor(1))
  }
}

/// A layer that applies layer normalization over a mini-batch of inputs.
///
/// Reference: [Layer Normalization](https://arxiv.org/abs/1607.06450).
@frozen
public struct LayerNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The axis.
  @noDerivative public let axis: Int
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar

  /// Creates a layer normalization layer.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int,
    epsilon: Scalar
  ) {
    precondition(offset.rank == 1, "The offset must have rank 1.")
    precondition(scale.rank == 1, "The scale must have rank 1.")
    precondition(
      offset.shape == scale.shape,
      "The offset and the scale must have same shape.")
    self.offset = offset
    self.scale = scale
    self.axis = axis
    self.epsilon = epsilon
  }

  /// Creates a layer normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis that should be normalized.
  ///   - epsilon: The small scalar added to variance.
  public init(
    featureCount: Int,
    axis: Int,
    epsilon: Scalar = 0.001
  ) {
    self.init(
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      axis: axis,
      epsilon: epsilon)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable(reverse)
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    // If you need to block gradients for `epsilon`, call `stopGradient`.
    // (See below for the definition of `stopGradient(_:)` if not already defined.)
    let epsilon = Tensor<Scalar>(self.epsilon, deviceAndPrecisionLike: input).stoppedGradient()
    

    let positiveAxis = (input.rank + axis) % input.rank
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "The number of features of the input and the offset doesn't match."
    )

    // Reshape 'offset' and 'scale' for broadcasting along 'positiveAxis'.
    var broadcastShape = TensorShape(Array(repeating: 1, count: input.rank))
    broadcastShape[positiveAxis] = input.shape[positiveAxis]
    let offset = self.offset.reshaped(to: broadcastShape)
    let scale = self.scale.reshaped(to: broadcastShape)

    // Compute mean and variance along the specified axis.
    let moments = input.moments(alongAxes: positiveAxis)
    // Normalize with variance epsilon, multiplied by 'scale'.
    let inv = rsqrt(moments.variance + epsilon) * scale

    // The final batch-normalization formula.
    return (input - moments.mean) * inv + offset
  }

}

/// A layer that applies group normalization over a mini-batch of inputs.
///
/// Reference: [Group Normalization](https://arxiv.org/abs/1803.08494).
@frozen
public struct GroupNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The number of groups.
  @noDerivative public let groupCount: Int
  /// The axis where the features lie.
  @noDerivative public let axis: Int
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar

  /// Creates a group normalization layer.
  /// - Parameters:
  ///   - offset: The initial offset value.
  ///   - scale: The initial scale value.
  ///   - groupCount: The number of groups.
  ///   - axis: The axis where the features lie.
  ///   - epsilon: The variance epsilon value.
  /// - Precondition: The axis cannot be batch axis.
  /// - Precondition: The offset must have rank 1.
  /// - Precondition: The number of elements of the offset must be divisible by groups.
  /// - Precondition: The offset and the scale must have same shape.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    groupCount: Int,
    axis: Int,
    epsilon: Scalar
  ) {
    precondition(axis != 0, "The axis cannot be batch axis.")
    precondition(offset.rank == 1, "The offset must have rank 1.")
    precondition(
      offset.shape[0].isMultiple(of: groupCount),
      "The number of elements of the offset must be divisible by the group count.")
    precondition(
      offset.shape == scale.shape,
      "The offset and the scale must have same shape.")
    self.offset = offset
    self.scale = scale
    self.groupCount = groupCount
    self.axis = axis
    self.epsilon = epsilon
  }

  /// Creates a group normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - groupCount: The number of groups.
  ///   - axis: The axis where the features lie. The default value is -1.
  ///   - epsilon: The small scalar added to variance. The default value is 0.001.
  /// - Precondition: The axis cannot be batch axis.
  /// - Precondition: The feature count must be divisible by groups.
  public init(
    featureCount: Int,
    groupCount: Int,
    axis: Int = -1,
    epsilon: Scalar = 1e-3
  ) {
    precondition(
      featureCount.isMultiple(of: groupCount),
      "The feature count must be divisible by groups.")
    self.init(
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      groupCount: groupCount,
      axis: axis,
      epsilon: epsilon
    )
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  /// - Precondition: The axis cannot be batch axis.
  /// - Precondition: The numbers of features of the input and the offset must be same.
  @differentiable(reverse)
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(positiveAxis != 0, "The axis cannot be batch axis.")
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "The numbers of features of the input and the offset must be the same."
    )

    // Prepare offset and scale for broadcasting along `groupCount`.
    var offset = self.offset
    var scale = self.scale
    var broadcastShape = TensorShape([Int](repeating: 1, count: input.rank + 1))
    broadcastShape[positiveAxis] = groupCount
    broadcastShape[positiveAxis + 1] = input.shape[positiveAxis] / groupCount
    offset = offset.reshaped(to: broadcastShape)
    scale = scale.reshaped(to: broadcastShape)

    // Reshape input so that its channel dimension is split into [groupCount, remainder].
    var groupShape = input.shape
    groupShape[positiveAxis] /= groupCount
    groupShape.insert(groupCount, at: positiveAxis)
    let grouped = input.reshaped(to: groupShape)

    // We will compute mean and variance along all axes except the newly inserted one
    // (plus the leading axis for the batch).
    var normalizedAxes = Array(1..<grouped.rank)
    normalizedAxes.remove(at: positiveAxis - 1)
    let moments = grouped.moments(alongAxes: normalizedAxes)

    // Replace `withoutDerivative` with `stoppedGradient()`.
    let eps = Tensor<Scalar>(self.epsilon, deviceAndPrecisionLike: input).stoppedGradient()

    // Normalize.
    let normalized = normalize(
      grouped,
      mean: moments.mean, 
      variance: moments.variance,
      offset: offset,
      scale: scale,
      varianceEpsilon: eps
    )
    return normalized.reshaped(to: input.shape)
  }
}

/// A layer that applies instance normalization over a mini-batch of inputs.
///
/// Reference: [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).
@frozen
public struct InstanceNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The general normalization layer of which `self` is a special case.
  var delegate: GroupNorm<Scalar>

  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar> {
    _read { yield delegate.offset }
    _modify { yield &delegate.offset }
  }

  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar> {
    _read { yield delegate.scale }
    _modify { yield &delegate.scale }
  }

  /// The axis where the features lie.
  public var axis: Int { delegate.axis }
  /// The variance epsilon value.
  public var epsilon: Scalar { delegate.epsilon }

  /// Creates a instance normalization layer.
  /// - Parameters:
  ///   - offset: The initial offset value.
  ///   - scale: The initial scale value.
  ///   - axis: The axis where the features lie.
  ///   - epsilon: The variance epsilon value.
  /// - Precondition: The axis cannot be batch axis.
  /// - Precondition: The offset must have rank 1.
  /// - Precondition: The offset and the scale must have same shape.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int,
    epsilon: Scalar
  ) {
    delegate = GroupNorm(
      offset: offset,
      scale: scale,
      groupCount: offset.shape[0],
      axis: axis,
      epsilon: epsilon)
  }

  /// Creates a instance normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis where the features lie. The default value is -1.
  ///   - epsilon: The small scalar added to variance. The default value is 0.001.
  /// - Precondition: The axis cannot be batch axis.
  /// - Precondition: The numbers of features of the input and the offset must be same.
  public init(
    featureCount: Int,
    axis: Int = -1,
    epsilon: Scalar = 1e-3
  ) {
    delegate = GroupNorm(
      featureCount: featureCount,
      groupCount: featureCount,
      axis: axis,
      epsilon: epsilon)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable(reverse)
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    delegate(input)
  }
}
