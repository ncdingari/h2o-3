// Generated by the protocol buffer compiler.  DO NOT EDIT!

package hex.genmodel.algos.deepwater.caffe.nano;

@SuppressWarnings("hiding")
public interface Deepwater {

  // enum Type
  public static final int Create = 0;
  public static final int Train = 1;
  public static final int Predict = 2;
  public static final int SaveGraph = 3;
  public static final int Save = 4;
  public static final int Load = 5;
  public static final int Success = 10;
  public static final int Failure = 11;

  public static final class Cmd extends
      com.google.protobuf.nano.MessageNano {

    private static volatile Cmd[] _emptyArray;
    public static Cmd[] emptyArray() {
      // Lazily initializes the empty array
      if (_emptyArray == null) {
        synchronized (
            com.google.protobuf.nano.InternalNano.LAZY_INIT_LOCK) {
          if (_emptyArray == null) {
            _emptyArray = new Cmd[0];
          }
        }
      }
      return _emptyArray;
    }

    // .deepwater.Type type = 1;
    public int type;

    // string graph = 100;
    public java.lang.String graph;

    // repeated int32 input_shape = 101;
    public int[] inputShape;

    // string solver_type = 102;
    public java.lang.String solverType;

    // float learning_rate = 103;
    public float learningRate;

    // float momentum = 104;
    public float momentum;

    // int64 random_seed = 105;
    public long randomSeed;

    // bool use_gpu = 106;
    public boolean useGpu;

    // bool regression = 107;
    public boolean regression;

    // repeated int32 sizes = 201;
    public int[] sizes;

    // repeated string types = 202;
    public java.lang.String[] types;

    // repeated double dropout_ratios = 203;
    public double[] dropoutRatios;

    // repeated bytes data = 300;
    public byte[][] data;

    // string path = 400;
    public java.lang.String path;

    public Cmd() {
      clear();
    }

    public Cmd clear() {
      type = hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Create;
      graph = "";
      inputShape = com.google.protobuf.nano.WireFormatNano.EMPTY_INT_ARRAY;
      solverType = "";
      learningRate = 0F;
      momentum = 0F;
      randomSeed = 0L;
      useGpu = false;
      regression = false;
      sizes = com.google.protobuf.nano.WireFormatNano.EMPTY_INT_ARRAY;
      types = com.google.protobuf.nano.WireFormatNano.EMPTY_STRING_ARRAY;
      dropoutRatios = com.google.protobuf.nano.WireFormatNano.EMPTY_DOUBLE_ARRAY;
      data = com.google.protobuf.nano.WireFormatNano.EMPTY_BYTES_ARRAY;
      path = "";
      cachedSize = -1;
      return this;
    }

    @Override
    public void writeTo(com.google.protobuf.nano.CodedOutputByteBufferNano output)
        throws java.io.IOException {
      if (this.type != hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Create) {
        output.writeInt32(1, this.type);
      }
      if (!this.graph.equals("")) {
        output.writeString(100, this.graph);
      }
      if (this.inputShape != null && this.inputShape.length > 0) {
        for (int i = 0; i < this.inputShape.length; i++) {
          output.writeInt32(101, this.inputShape[i]);
        }
      }
      if (!this.solverType.equals("")) {
        output.writeString(102, this.solverType);
      }
      if (java.lang.Float.floatToIntBits(this.learningRate)
          != java.lang.Float.floatToIntBits(0F)) {
        output.writeFloat(103, this.learningRate);
      }
      if (java.lang.Float.floatToIntBits(this.momentum)
          != java.lang.Float.floatToIntBits(0F)) {
        output.writeFloat(104, this.momentum);
      }
      if (this.randomSeed != 0L) {
        output.writeInt64(105, this.randomSeed);
      }
      if (this.useGpu != false) {
        output.writeBool(106, this.useGpu);
      }
      if (this.regression != false) {
        output.writeBool(107, this.regression);
      }
      if (this.sizes != null && this.sizes.length > 0) {
        for (int i = 0; i < this.sizes.length; i++) {
          output.writeInt32(201, this.sizes[i]);
        }
      }
      if (this.types != null && this.types.length > 0) {
        for (int i = 0; i < this.types.length; i++) {
          java.lang.String element = this.types[i];
          if (element != null) {
            output.writeString(202, element);
          }
        }
      }
      if (this.dropoutRatios != null && this.dropoutRatios.length > 0) {
        for (int i = 0; i < this.dropoutRatios.length; i++) {
          output.writeDouble(203, this.dropoutRatios[i]);
        }
      }
      if (this.data != null && this.data.length > 0) {
        for (int i = 0; i < this.data.length; i++) {
          byte[] element = this.data[i];
          if (element != null) {
            output.writeBytes(300, element);
          }
        }
      }
      if (!this.path.equals("")) {
        output.writeString(400, this.path);
      }
      super.writeTo(output);
    }

    @Override
    protected int computeSerializedSize() {
      int size = super.computeSerializedSize();
      if (this.type != hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Create) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
          .computeInt32Size(1, this.type);
      }
      if (!this.graph.equals("")) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeStringSize(100, this.graph);
      }
      if (this.inputShape != null && this.inputShape.length > 0) {
        int dataSize = 0;
        for (int i = 0; i < this.inputShape.length; i++) {
          int element = this.inputShape[i];
          dataSize += com.google.protobuf.nano.CodedOutputByteBufferNano
              .computeInt32SizeNoTag(element);
        }
        size += dataSize;
        size += 2 * this.inputShape.length;
      }
      if (!this.solverType.equals("")) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeStringSize(102, this.solverType);
      }
      if (java.lang.Float.floatToIntBits(this.learningRate)
          != java.lang.Float.floatToIntBits(0F)) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeFloatSize(103, this.learningRate);
      }
      if (java.lang.Float.floatToIntBits(this.momentum)
          != java.lang.Float.floatToIntBits(0F)) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeFloatSize(104, this.momentum);
      }
      if (this.randomSeed != 0L) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeInt64Size(105, this.randomSeed);
      }
      if (this.useGpu != false) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeBoolSize(106, this.useGpu);
      }
      if (this.regression != false) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeBoolSize(107, this.regression);
      }
      if (this.sizes != null && this.sizes.length > 0) {
        int dataSize = 0;
        for (int i = 0; i < this.sizes.length; i++) {
          int element = this.sizes[i];
          dataSize += com.google.protobuf.nano.CodedOutputByteBufferNano
              .computeInt32SizeNoTag(element);
        }
        size += dataSize;
        size += 2 * this.sizes.length;
      }
      if (this.types != null && this.types.length > 0) {
        int dataCount = 0;
        int dataSize = 0;
        for (int i = 0; i < this.types.length; i++) {
          java.lang.String element = this.types[i];
          if (element != null) {
            dataCount++;
            dataSize += com.google.protobuf.nano.CodedOutputByteBufferNano
                .computeStringSizeNoTag(element);
          }
        }
        size += dataSize;
        size += 2 * dataCount;
      }
      if (this.dropoutRatios != null && this.dropoutRatios.length > 0) {
        int dataSize = 8 * this.dropoutRatios.length;
        size += dataSize;
        size += 2 * this.dropoutRatios.length;
      }
      if (this.data != null && this.data.length > 0) {
        int dataCount = 0;
        int dataSize = 0;
        for (int i = 0; i < this.data.length; i++) {
          byte[] element = this.data[i];
          if (element != null) {
            dataCount++;
            dataSize += com.google.protobuf.nano.CodedOutputByteBufferNano
                .computeBytesSizeNoTag(element);
          }
        }
        size += dataSize;
        size += 2 * dataCount;
      }
      if (!this.path.equals("")) {
        size += com.google.protobuf.nano.CodedOutputByteBufferNano
            .computeStringSize(400, this.path);
      }
      return size;
    }

    @Override
    public Cmd mergeFrom(
            com.google.protobuf.nano.CodedInputByteBufferNano input)
        throws java.io.IOException {
      while (true) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            return this;
          default: {
            if (!com.google.protobuf.nano.WireFormatNano.parseUnknownField(input, tag)) {
              return this;
            }
            break;
          }
          case 8: {
            int value = input.readInt32();
            switch (value) {
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Create:
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Train:
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Predict:
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.SaveGraph:
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Save:
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Load:
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Success:
              case hex.genmodel.algos.deepwater.caffe.nano.Deepwater.Failure:
                this.type = value;
                break;
            }
            break;
          }
          case 802: {
            this.graph = input.readString();
            break;
          }
          case 808: {
            int arrayLength = com.google.protobuf.nano.WireFormatNano
                .getRepeatedFieldArrayLength(input, 808);
            int i = this.inputShape == null ? 0 : this.inputShape.length;
            int[] newArray = new int[i + arrayLength];
            if (i != 0) {
              java.lang.System.arraycopy(this.inputShape, 0, newArray, 0, i);
            }
            for (; i < newArray.length - 1; i++) {
              newArray[i] = input.readInt32();
              input.readTag();
            }
            // Last one without readTag.
            newArray[i] = input.readInt32();
            this.inputShape = newArray;
            break;
          }
          case 810: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            // First pass to compute array length.
            int arrayLength = 0;
            int startPos = input.getPosition();
            while (input.getBytesUntilLimit() > 0) {
              input.readInt32();
              arrayLength++;
            }
            input.rewindToPosition(startPos);
            int i = this.inputShape == null ? 0 : this.inputShape.length;
            int[] newArray = new int[i + arrayLength];
            if (i != 0) {
              java.lang.System.arraycopy(this.inputShape, 0, newArray, 0, i);
            }
            for (; i < newArray.length; i++) {
              newArray[i] = input.readInt32();
            }
            this.inputShape = newArray;
            input.popLimit(limit);
            break;
          }
          case 818: {
            this.solverType = input.readString();
            break;
          }
          case 829: {
            this.learningRate = input.readFloat();
            break;
          }
          case 837: {
            this.momentum = input.readFloat();
            break;
          }
          case 840: {
            this.randomSeed = input.readInt64();
            break;
          }
          case 848: {
            this.useGpu = input.readBool();
            break;
          }
          case 856: {
            this.regression = input.readBool();
            break;
          }
          case 1608: {
            int arrayLength = com.google.protobuf.nano.WireFormatNano
                .getRepeatedFieldArrayLength(input, 1608);
            int i = this.sizes == null ? 0 : this.sizes.length;
            int[] newArray = new int[i + arrayLength];
            if (i != 0) {
              java.lang.System.arraycopy(this.sizes, 0, newArray, 0, i);
            }
            for (; i < newArray.length - 1; i++) {
              newArray[i] = input.readInt32();
              input.readTag();
            }
            // Last one without readTag.
            newArray[i] = input.readInt32();
            this.sizes = newArray;
            break;
          }
          case 1610: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            // First pass to compute array length.
            int arrayLength = 0;
            int startPos = input.getPosition();
            while (input.getBytesUntilLimit() > 0) {
              input.readInt32();
              arrayLength++;
            }
            input.rewindToPosition(startPos);
            int i = this.sizes == null ? 0 : this.sizes.length;
            int[] newArray = new int[i + arrayLength];
            if (i != 0) {
              java.lang.System.arraycopy(this.sizes, 0, newArray, 0, i);
            }
            for (; i < newArray.length; i++) {
              newArray[i] = input.readInt32();
            }
            this.sizes = newArray;
            input.popLimit(limit);
            break;
          }
          case 1618: {
            int arrayLength = com.google.protobuf.nano.WireFormatNano
                .getRepeatedFieldArrayLength(input, 1618);
            int i = this.types == null ? 0 : this.types.length;
            java.lang.String[] newArray = new java.lang.String[i + arrayLength];
            if (i != 0) {
              java.lang.System.arraycopy(this.types, 0, newArray, 0, i);
            }
            for (; i < newArray.length - 1; i++) {
              newArray[i] = input.readString();
              input.readTag();
            }
            // Last one without readTag.
            newArray[i] = input.readString();
            this.types = newArray;
            break;
          }
          case 1625: {
            int arrayLength = com.google.protobuf.nano.WireFormatNano
                .getRepeatedFieldArrayLength(input, 1625);
            int i = this.dropoutRatios == null ? 0 : this.dropoutRatios.length;
            double[] newArray = new double[i + arrayLength];
            if (i != 0) {
              java.lang.System.arraycopy(this.dropoutRatios, 0, newArray, 0, i);
            }
            for (; i < newArray.length - 1; i++) {
              newArray[i] = input.readDouble();
              input.readTag();
            }
            // Last one without readTag.
            newArray[i] = input.readDouble();
            this.dropoutRatios = newArray;
            break;
          }
          case 1626: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            int arrayLength = length / 8;
            int i = this.dropoutRatios == null ? 0 : this.dropoutRatios.length;
            double[] newArray = new double[i + arrayLength];
            if (i != 0) {
              java.lang.System.arraycopy(this.dropoutRatios, 0, newArray, 0, i);
            }
            for (; i < newArray.length; i++) {
              newArray[i] = input.readDouble();
            }
            this.dropoutRatios = newArray;
            input.popLimit(limit);
            break;
          }
          case 2402: {
            int arrayLength = com.google.protobuf.nano.WireFormatNano
                .getRepeatedFieldArrayLength(input, 2402);
            int i = this.data == null ? 0 : this.data.length;
            byte[][] newArray = new byte[i + arrayLength][];
            if (i != 0) {
              java.lang.System.arraycopy(this.data, 0, newArray, 0, i);
            }
            for (; i < newArray.length - 1; i++) {
              newArray[i] = input.readBytes();
              input.readTag();
            }
            // Last one without readTag.
            newArray[i] = input.readBytes();
            this.data = newArray;
            break;
          }
          case 3202: {
            this.path = input.readString();
            break;
          }
        }
      }
    }

    public static Cmd parseFrom(byte[] data)
        throws com.google.protobuf.nano.InvalidProtocolBufferNanoException {
      return com.google.protobuf.nano.MessageNano.mergeFrom(new Cmd(), data);
    }

    public static Cmd parseFrom(
            com.google.protobuf.nano.CodedInputByteBufferNano input)
        throws java.io.IOException {
      return new Cmd().mergeFrom(input);
    }
  }
}