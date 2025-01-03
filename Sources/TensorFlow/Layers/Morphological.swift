import _Differentiation

/// A 2-D morphological dilation layer.
///
/// This layer returns the morphological dilation of the input tensor with the provided filters.
@frozen
public struct Dilation2D<Scalar: TensorFlowFloatingPoint>: Layer {
  // Conform to `Layer`:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  /// The 4-D dilation filter of shape [filter height, filter width, input channels, output channels].
  public var filter: Tensor<Scalar>
  /// The strides of the sliding window for spatial dimensions, i.e. (stride height, stride width).
  @noDerivative public let strides: (Int, Int)
  /// The padding algorithm for dilation.
  @noDerivative public let padding: Padding
  /// The dilation factor for spatial dimensions, i.e. (dilation height, dilation width).
  @noDerivative public let rates: (Int, Int)

  /// Creates a `Dilation2D` layer with the specified filter, strides, dilations, and padding.
  ///
  /// - Parameters:
  ///   - filter: The 4-D dilation filter of shape
  ///       [filter height, filter width, input channels, output channels].
  ///   - strides: The strides of the sliding window for spatial dimensions, (height, width).
  ///   - rates: The dilation rates for spatial dimensions, (height, width).
  ///   - padding: The padding algorithm for dilation (e.g. `.valid` or `.same`).
  public init(
    filter: Tensor<Scalar>,
    strides: (Int, Int) = (1, 1),
    rates: (Int, Int) = (1, 1),
    padding: Padding = .valid
  ) {
    self.filter = filter
    self.strides = strides
    self.padding = padding
    self.rates = rates
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// The output shape is determined by `strides`, `rates`, and `padding`.
  /// - Parameter input: The input of shape [batch size, input height, input width, input channels].
  /// - Returns: The output of shape [batch size, output height, output width, output channels].
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    // Perform morphological dilation using `dilation2D`.
    dilation2D(
      input,
      filter: filter,
      strides: (1, strides.0, strides.1, 1),
      rates: (1, rates.0, rates.1, 1),
      padding: padding
    )
  }

  /// Satisfies the `Layer` requirement.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}

/// A 2-D morphological erosion layer.
///
/// This layer returns the morphological erosion of the input tensor with the provided filters.
@frozen
public struct Erosion2D<Scalar: TensorFlowFloatingPoint>: Layer {
  // Conform to `Layer`:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  /// The 4-D erosion filter of shape [filter height, filter width, input channels, output channels].
  public var filter: Tensor<Scalar>
  /// The strides of the sliding window for spatial dimensions, i.e. (stride height, stride width).
  @noDerivative public let strides: (Int, Int)
  /// The padding algorithm for erosion.
  @noDerivative public let padding: Padding
  /// The dilation factor for spatial dimensions, i.e. (dilation height, dilation width).
  @noDerivative public let rates: (Int, Int)

  /// Creates an `Erosion2D` layer with the specified filter, strides, dilations, and padding.
  ///
  /// - Parameters:
  ///   - filter: The 4-D erosion filter of shape
  ///       [filter height, filter width, input channels, output channels].
  ///   - strides: The strides of the sliding window for spatial dimensions, (height, width).
  ///   - rates: The dilation rates for spatial dimensions, (height, width).
  ///   - padding: The padding algorithm for erosion (e.g. `.valid` or `.same`).
  public init(
    filter: Tensor<Scalar>,
    strides: (Int, Int) = (1, 1),
    rates: (Int, Int) = (1, 1),
    padding: Padding = .valid
  ) {
    self.filter = filter
    self.strides = strides
    self.padding = padding
    self.rates = rates
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// The output shape is determined by `strides`, `rates`, and `padding`.
  /// - Parameter input: The input of shape [batch size, height, width, channels].
  /// - Returns: The output of shape [batch size, outHeight, outWidth, outChannels].
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    // Perform morphological erosion using `erosion2D`.
    erosion2D(
      input,
      filter: filter,
      strides: (1, strides.0, strides.1, 1),
      rates: (1, rates.0, rates.1, 1),
      padding: padding
    )
  }

  /// Satisfies the `Layer` requirement.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}
