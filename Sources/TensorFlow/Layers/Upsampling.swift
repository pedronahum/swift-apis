import _Differentiation

/// An upsampling layer for 1-D inputs.
@frozen
public struct UpSampling1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  // Conform to ParameterlessLayer:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  // No parameters, so move(by:) is a no-op.
  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let size: Int

  /// Creates an upsampling layer.
  ///
  /// - Parameter size: The upsampling factor for timesteps.
  public init(size: Int) {
    self.size = size
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let shape = input.shape
    let (batchSize, timesteps, channels) = (shape[0], shape[1], shape[2])
    // Expand the middle dimension by `size`, multiply by an all-ones tensor,
    // then reshape back. This is effectively "repeat" along axis=2, but done with broadcasting.
    let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1], on: input.device)
    let upSampling = input.reshaped(to: [batchSize, timesteps, 1, channels]) * scaleOnes
    return upSampling.reshaped(to: [batchSize, timesteps * size, channels])
  }

  /// Satisfies the `ParameterlessLayer` requirement.
  /// Delegates to `forward(_:)`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

/// An upsampling layer for 2-D inputs.
@frozen
public struct UpSampling2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let size: Int

  /// Creates an upsampling layer.
  ///
  /// - Parameter size: The upsampling factor for rows and columns.
  public init(size: Int) {
    self.size = size
  }

  /// The main "forward" pass logic.
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let device = input.device
    let shape = input.shape
    let (batchSize, height, width, channels) = (shape[0], shape[1], shape[2], shape[3])
    // Expand the 2D dims by `size`, multiply by an all-ones tensor, then reshape back.
    let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1], on: device)
    let upSampling = input.reshaped(to: [batchSize, height, 1, width, 1, channels]) * scaleOnes
    return upSampling.reshaped(to: [batchSize, height * size, width * size, channels])
  }

  /// Satisfies `ParameterlessLayer` by delegating to `forward`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

/// An upsampling layer for 3-D inputs.
@frozen
public struct UpSampling3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  // Conform to ParameterlessLayer:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) {}

  @noDerivative public let size: Int

  /// Creates an upsampling layer.
  ///
  /// - Parameter size: The upsampling factor for the 3D spatial dims.
  public init(size: Int) {
    self.size = size
  }

  // We define both forward(_:) and callAsFunction(_:) with the correct attribute.

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    // Repeat along axis=1,2,3 by factor 'size'.
    var result = repeatingElements(input, alongAxis: 1, count: size)
    result = repeatingElements(result, alongAxis: 2, count: size)
    result = repeatingElements(result, alongAxis: 3, count: size)
    return result
  }

  @differentiable(reverse)
  private func repeatingElements(
    _ input: Tensor<Scalar>,
    alongAxis axis: Int,
    count: Int
  ) -> Tensor<Scalar> {
    let splits = _Raw.split(
      splitDim: Tensor<Int32>(Int32(axis), on: input.device),
      value: input,
      numSplit: Int64(input.shape[axis])
    )
    let repeated = splits.flatMap { x in Array(repeating: x, count: count) }
    return Tensor<Scalar>(concatenating: repeated, alongAxis: axis)
  }

  @derivative(of: repeatingElements)
  private func _vjpRepeatingElements(
    _ input: Tensor<Scalar>,
    alongAxis axis: Int,
    count: Int
  ) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (TangentVector, Tensor<Scalar>)) {
    let value = repeatingElements(input, alongAxis: axis, count: count)
    return (
      value,
      { v in
        // We "undo" repeating by summing the relevant slices.
        let splits = _Raw.split(
          splitDim: Tensor<Int32>(Int32(axis), on: v.device),
          value: v,
          numSplit: Int64(input.shape[axis])
        )
        let summed = splits.map { x in x.sum(alongAxes: axis) }
        let concatenated = Tensor<Scalar>(concatenating: summed, alongAxis: axis)
        return (.zero, concatenated)
      }
    )
  }
}
