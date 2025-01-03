import _Differentiation

// MARK: - Conv1D

/// A 1-D convolution layer (e.g. temporal convolution over a time-series).
@frozen
public struct Conv1D<Scalar: TensorFlowFloatingPoint>: Layer {
  // Protocol conformances:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  // Stored properties:
  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let stride: Int
  @noDerivative public let padding: Padding
  @noDerivative public let dilation: Int
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  // Initializers:
  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    stride: Int = 1,
    padding: Padding = .valid,
    dilation: Int = 1
  ) {
    self.filter = filter
    self.bias = bias ?? .zero
    self.activation = activation
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.useBias = (bias != nil)
  }

  public init(
    filterShape: (Int, Int, Int),
    stride: Int = 1,
    padding: Padding = .valid,
    dilation: Int = 1,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let shape = TensorShape([filterShape.0, filterShape.1, filterShape.2])
    self.init(
      filter: filterInitializer(shape),
      bias: useBias ? biasInitializer([filterShape.2]) : nil,
      activation: activation,
      stride: stride,
      padding: padding,
      dilation: dilation
    )
  }

  // Forward + callAsFunction:
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let conv = conv1D(
      input, filter: filter,
      stride: stride,
      padding: padding,
      dilation: dilation
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

// MARK: - Conv2D

/// A 2-D convolution layer (spatial convolution over images).
@frozen
public struct Conv2D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int)
  @noDerivative public let padding: Padding
  @noDerivative public let dilations: (Int, Int)
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1)
  ) {
    self.filter = filter
    self.bias = bias ?? .zero
    self.activation = activation
    self.strides = strides
    self.padding = padding
    self.dilations = dilations
    self.useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let conv = conv2D(
      input,
      filter: filter,
      strides: (1, strides.0, strides.1, 1),
      padding: padding,
      dilations: (1, dilations.0, dilations.1, 1)
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension Conv2D {
  public init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1),
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let filterTensorShape = TensorShape([
      filterShape.0, filterShape.1, filterShape.2, filterShape.3
    ])
    self.init(
      filter: filterInitializer(filterTensorShape),
      bias: useBias ? biasInitializer([filterShape.3]) : nil,
      activation: activation,
      strides: strides,
      padding: padding,
      dilations: dilations
    )
  }
}

// MARK: - Conv3D

@frozen
public struct Conv3D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int, Int)
  @noDerivative public let padding: Padding
  @noDerivative public let dilations: (Int, Int, Int)
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    strides: (Int, Int, Int) = (1, 1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int, Int) = (1, 1, 1)
  ) {
    self.filter = filter
    self.bias = bias ?? .zero
    self.activation = activation
    self.strides = strides
    self.padding = padding
    self.dilations = dilations
    self.useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let conv = conv3D(
      input,
      filter: filter,
      strides: (1, strides.0, strides.1, strides.2, 1),
      padding: padding,
      dilations: (1, dilations.0, dilations.1, dilations.2, 1)
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension Conv3D {
  public init(
    filterShape: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int) = (1, 1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int, Int) = (1, 1, 1),
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let filterTensorShape = TensorShape([
      filterShape.0, filterShape.1, filterShape.2, filterShape.3, filterShape.4
    ])
    self.init(
      filter: filterInitializer(filterTensorShape),
      bias: useBias ? biasInitializer([filterShape.4]) : nil,
      activation: activation,
      strides: strides,
      padding: padding,
      dilations: dilations
    )
  }
}

// MARK: - TransposedConv1D

@frozen
public struct TransposedConv1D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let stride: Int
  @noDerivative public let padding: Padding
  @noDerivative public let paddingIndex: Int
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    stride: Int = 1,
    padding: Padding = .valid
  ) {
    self.filter = filter
    self.bias = bias ?? .zero
    self.activation = activation
    self.stride = stride
    self.padding = padding
    self.paddingIndex = padding == .same ? 0 : 1
    useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let batchSize = input.shape[0]
    let w = (input.shape[1] - (1 * paddingIndex)) * stride + (filter.shape[0] * paddingIndex)
    let c = filter.shape[2]
    let newShape = [Int64(batchSize), 1, Int64(w), Int64(c)]
    let conv = transposedConv2D(
      input.expandingShape(at: 1),
      shape: newShape,
      filter: filter.expandingShape(at: 0),
      strides: (1, 1, stride, 1),
      padding: padding
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension TransposedConv1D {
  public init(
    filterShape: (Int, Int, Int),
    stride: Int = 1,
    padding: Padding = .valid,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let filterTensorShape = TensorShape([
      filterShape.0, filterShape.1, filterShape.2
    ])
    self.init(
      filter: filterInitializer(filterTensorShape),
      bias: useBias ? biasInitializer([filterShape.2]) : nil,
      activation: activation,
      stride: stride,
      padding: padding
    )
  }
}

// MARK: - TransposedConv2D

@frozen
public struct TransposedConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int)
  @noDerivative public let padding: Padding
  @noDerivative public let paddingIndex: Int
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid
  ) {
    self.filter = filter
    self.bias = bias ?? .zero
    self.activation = activation
    self.strides = strides
    self.padding = padding
    self.paddingIndex = (padding == .same) ? 0 : 1
    self.useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let batchSize = input.shape[0]
    let h = (input.shape[1] - (1 * paddingIndex)) * strides.0 + (filter.shape[0] * paddingIndex)
    let w = (input.shape[2] - (1 * paddingIndex)) * strides.1 + (filter.shape[1] * paddingIndex)
    let c = filter.shape[2]
    let newShape = [Int64(batchSize), Int64(h), Int64(w), Int64(c)]
    let conv = transposedConv2D(
      input,
      shape: newShape,
      filter: filter,
      strides: (1, strides.0, strides.1, 1),
      padding: padding
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension TransposedConv2D {
  public init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let filterTensorShape = TensorShape([
      filterShape.0, filterShape.1, filterShape.2, filterShape.3
    ])
    self.init(
      filter: filterInitializer(filterTensorShape),
      bias: useBias ? biasInitializer([filterShape.2]) : nil,
      activation: activation,
      strides: strides,
      padding: padding
    )
  }
}

// MARK: - TransposedConv3D

@frozen
public struct TransposedConv3D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int, Int)
  @noDerivative public let padding: Padding
  @noDerivative public let paddingIndex: Int
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    strides: (Int, Int, Int) = (1, 1, 1),
    padding: Padding = .valid
  ) {
    self.filter = filter
    self.bias = bias ?? .zero
    self.activation = activation
    self.strides = strides
    self.padding = padding
    self.paddingIndex = padding == .same ? 0 : 1
    self.useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let batchSize = input.shape[0]
    let w = (input.shape[1] - (1 * paddingIndex)) * strides.0 + (filter.shape[0] * paddingIndex)
    let h = (input.shape[2] - (1 * paddingIndex)) * strides.1 + (filter.shape[1] * paddingIndex)
    let d = (input.shape[3] - (1 * paddingIndex)) * strides.2 + (filter.shape[2] * paddingIndex)
    let c = filter.shape[3]
    let newShape = Tensor<Int32>([Int32(batchSize), Int32(w), Int32(h), Int32(d), Int32(c)], on: input.device)
    let conv = conv3DBackpropInput(
      input,
      shape: newShape,
      filter: filter,
      strides: (1, strides.0, strides.1, strides.2, 1),
      padding: padding
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension TransposedConv3D {
  public init(
    filterShape: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int) = (1, 1, 1),
    padding: Padding = .valid,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let filterTensorShape = TensorShape([
      filterShape.0, filterShape.1, filterShape.2, filterShape.3, filterShape.4
    ])
    self.init(
      filter: filterInitializer(filterTensorShape),
      bias: useBias ? biasInitializer([filterShape.4]) : nil,
      activation: activation,
      strides: strides,
      padding: padding
    )
  }
}

// MARK: - DepthwiseConv2D

@frozen
public struct DepthwiseConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int)
  @noDerivative public let padding: Padding
  @noDerivative public let dilations: (Int, Int)
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1)
  ) {
    self.filter = filter
    self.bias = bias ?? .zero
    self.activation = activation
    self.strides = strides
    self.padding = padding
    self.dilations = dilations
    self.useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let conv = depthwiseConv2D(
      input,
      filter: filter,
      strides: (1, strides.0, strides.1, 1),
      padding: padding,
      dilations: (1, dilations.0, dilations.1, 1)
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension DepthwiseConv2D {
  public init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1),
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let filterTensorShape = TensorShape([
      filterShape.0, filterShape.1, filterShape.2, filterShape.3
    ])
    self.init(
      filter: filterInitializer(filterTensorShape),
      bias: useBias ? biasInitializer([filterShape.2 * filterShape.3]) : nil,
      activation: activation,
      strides: strides,
      padding: padding,
      dilations: dilations
    )
  }
}

// MARK: - ZeroPadding1D

public struct ZeroPadding1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  // Satisfy ParameterlessLayer:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) { /* no-op */ }

  @noDerivative public let padding: (Int, Int)

  public init(padding: (Int, Int)) {
    self.padding = padding
  }

  public init(padding: Int) {
    self.init(padding: (padding, padding))
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    input.padded(forSizes: [(0, 0), padding, (0, 0)])
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

// MARK: - ZeroPadding2D

public struct ZeroPadding2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) { /* no-op */ }

  @noDerivative public let padding: ((Int, Int), (Int, Int))

  public init(padding: ((Int, Int), (Int, Int))) {
    self.padding = padding
  }

  public init(padding: (Int, Int)) {
    self.init(padding: ((padding.0, padding.0), (padding.1, padding.1)))
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    input.padded(forSizes: [(0, 0), padding.0, padding.1, (0, 0)])
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

// MARK: - ZeroPadding3D

public struct ZeroPadding3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public typealias TangentVector = EmptyTangentVector

  public mutating func move(by offset: EmptyTangentVector) { /* no-op */ }

  @noDerivative public let padding: ((Int, Int), (Int, Int), (Int, Int))

  public init(padding: ((Int, Int), (Int, Int), (Int, Int))) {
    self.padding = padding
  }

  public init(padding: (Int, Int, Int)) {
    self.init(
      padding: (
        (padding.0, padding.0), 
        (padding.1, padding.1), 
        (padding.2, padding.2)
      )
    )
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    input.padded(forSizes: [(0, 0), padding.0, padding.1, padding.2, (0, 0)])
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

// MARK: - SeparableConv1D

@frozen
public struct SeparableConv1D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var depthwiseFilter: Tensor<Scalar>
  public var pointwiseFilter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let stride: Int
  @noDerivative public let padding: Padding
  @noDerivative public let dilation: Int
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    depthwiseFilter: Tensor<Scalar>,
    pointwiseFilter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    stride: Int = 1,
    padding: Padding = .valid,
    dilation: Int = 1
  ) {
    self.depthwiseFilter = depthwiseFilter
    self.pointwiseFilter = pointwiseFilter
    self.bias = bias ?? .zero
    self.activation = activation
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    // Depthwise uses a 2D operation with a trick:
    let depthwise = depthwiseConv2D(
      input.expandingShape(at: 1),
      filter: depthwiseFilter.expandingShape(at: 1),
      strides: (1, stride, stride, 1),
      padding: padding,
      dilations: (1, dilation, dilation, 1)
    )
    let pointwise = conv2D(
      depthwise,
      filter: pointwiseFilter.expandingShape(at: 1),
      strides: (1, 1, 1, 1),
      padding: padding,
      dilations: (1, 1, 1, 1)
    )
    let out = pointwise.squeezingShape(at: 1)
    return activation(useBias ? (out + bias) : out)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension SeparableConv1D {
  public init(
    depthwiseFilterShape: (Int, Int, Int),
    pointwiseFilterShape: (Int, Int, Int),
    stride: Int = 1,
    padding: Padding = .valid,
    dilation: Int = 1,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    depthwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    pointwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let depthwiseFilterTensorShape = TensorShape([
      depthwiseFilterShape.0, depthwiseFilterShape.1, depthwiseFilterShape.2
    ])
    let pointwiseFilterTensorShape = TensorShape([
      pointwiseFilterShape.0, pointwiseFilterShape.1, pointwiseFilterShape.2
    ])
    self.init(
      depthwiseFilter: depthwiseFilterInitializer(depthwiseFilterTensorShape),
      pointwiseFilter: pointwiseFilterInitializer(pointwiseFilterTensorShape),
      bias: useBias ? biasInitializer([pointwiseFilterShape.2]) : nil,
      activation: activation,
      stride: stride,
      padding: padding,
      dilation: dilation
    )
  }
}

// MARK: - SeparableConv2D

@frozen
public struct SeparableConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  public var depthwiseFilter: Tensor<Scalar>
  public var pointwiseFilter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int)
  @noDerivative public let padding: Padding
  @noDerivative public let dilations: (Int, Int)
  @noDerivative private let useBias: Bool

  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    depthwiseFilter: Tensor<Scalar>,
    pointwiseFilter: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation = identity,
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1)
  ) {
    self.depthwiseFilter = depthwiseFilter
    self.pointwiseFilter = pointwiseFilter
    self.bias = bias ?? .zero
    self.activation = activation
    self.strides = strides
    self.padding = padding
    self.dilations = dilations
    self.useBias = (bias != nil)
  }

  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let depthwise = depthwiseConv2D(
      input,
      filter: depthwiseFilter,
      strides: (1, strides.0, strides.1, 1),
      padding: padding,
      dilations: (1, dilations.0, dilations.1, 1)
    )
    let conv = conv2D(
      depthwise,
      filter: pointwiseFilter,
      strides: (1, 1, 1, 1),
      padding: padding,
      dilations: (1, 1, 1, 1)
    )
    return activation(useBias ? (conv + bias) : conv)
  }

  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    forward(input)
  }
}

extension SeparableConv2D {
  public init(
    depthwiseFilterShape: (Int, Int, Int, Int),
    pointwiseFilterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1),
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    depthwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    pointwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let depthwiseFilterTensorShape = TensorShape([
      depthwiseFilterShape.0, depthwiseFilterShape.1, depthwiseFilterShape.2, depthwiseFilterShape.3
    ])
    let pointwiseFilterTensorShape = TensorShape([
      pointwiseFilterShape.0, pointwiseFilterShape.1, pointwiseFilterShape.2, pointwiseFilterShape.3
    ])
    self.init(
      depthwiseFilter: depthwiseFilterInitializer(depthwiseFilterTensorShape),
      pointwiseFilter: pointwiseFilterInitializer(pointwiseFilterTensorShape),
      bias: useBias ? biasInitializer([pointwiseFilterShape.3]) : nil,
      activation: activation,
      strides: strides,
      padding: padding,
      dilations: dilations
    )
  }
}
