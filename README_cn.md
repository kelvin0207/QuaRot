# QuaRot 项目代码解析

## 数学逻辑
在论文的第四章 **4 Method（方法）** 中，作者详细阐述了 QuaRot 方案的具体实现流程 。整个方法分为两个大阶段：第一阶段（Stage 1）在全精度下对模型权重进行数学旋转，并在线插入少量的哈达玛操作 ；第二阶段（Stage 2）对调整后的权重进行量化，并加入激活值和缓存的在线量化操作 。

以下是对第四章各个关键步骤的深度解析：

### 阶段 1：模型旋转与变换 (Stage 1: Weight Modification & Rotation)

这一阶段的核心目标是利用第三章所述的**计算不变性**，在不改变模型前向传播输出的前提下 ，将模型内部的激活值和缓存转换到“无离群值”的旋转空间中 。

#### Stage 1a: 权重修改与全局旋转 (Weight Modification)

* **算子融合：** 首先，为了满足计算不变性的前置条件，作者将 LayerNorm 或 RMSNorm 的线性缩放参数（通常表示为对角矩阵 $\text{diag}(\alpha)$）提取出来，直接融合（Absorb）到紧邻的后方权重矩阵中 。
* **左乘/右乘哈达玛：** 选择一个与模型隐藏层维度相匹配的随机哈达玛矩阵 $Q$ 。
* 对于位于计算块**输入侧**的权重矩阵（如 Key 投影矩阵 $W_k$、Query 投影矩阵 $W_q$ 等），在其左侧乘以 $Q^{\top}$ ：


$$W_k \leftarrow Q^{\top} \text{diag}(\alpha) W_k$$

* 对于位于计算块**输出侧**的权重矩阵（如 $W_{\text{down}}$ 或 $W_{\text{out}}$），则在其右侧乘以 $Q$ 。
* **数学效果：** 通过这种修改，层与层之间流通的激活值矩阵实际上变成了 $X \leftarrow XQ$ 。如图 1 所示，这种全局旋转成功消除了中间激活值的一切离群值 。



#### Stage 1b: 旋转 FFN 内部激活值 (Rotate FFN activations)

虽然全局旋转 $Q$ 解决了层间激活值的问题，但 FFN 内部（在门控机制和激活函数操作之后，$W_{\text{down}}$ 之前）依然会产生新的离群值 。

* **引入在线哈达玛：** 作者在 $W_{\text{down}}$ 线性层之前，强行插入了一个全精度的在线哈达玛变换 $H$ 。
* **权重对冲融合：** 为了让这个插入的 $H$ 不改变前向传播结果，作者将对应的哈达玛矩阵融合进降级投影权重中，使其变为 $W_{\text{down}} \leftarrow HW_{\text{down}}$ 。结合 Stage 1a 的全局旋转，最终的 FFN 输出权重更新为 $HW_{\text{down}}Q$（如架构图 3 所示） 。


#### Stage 1c: 注意力 Value 投影旋转 (Attention Value Projection)

在自注意力模块中，每个头（Head）内部的 Value 矩阵 $V_h$ 与输出投影矩阵 $W_{\text{out}}^{(h)}$ 存在隐式的相乘关系 。为了平滑 Value 向量以允许其进行 4-bit 量化，作者利用了类似的技巧 ：

* **头内旋转（Head-wise）：** 使用符合每个头维度的哈达玛矩阵 $H_{d_h}$，对各头的权重进行变换 ：
$$W_v^{(h)} \leftarrow W_v^{(h)} H_{d_h}, \quad W_{\text{out}}^{(h)} \leftarrow H_{d_h} W_{\text{out}}^{(h)}$$

* **跨头共享与哈达玛变换：** 在矩阵拼接表达下，这相当于应用了克罗内克积结构 $(I \otimes H_{d_h})$ 。为了形成一个“完整”的哈达玛变换，作者利用了以下数学恒等式 ：

$$H_{n_h \times d_h} = (I \otimes H_{d_h})(H_{n_h} \otimes I)$$

因此，除了在权重中融合变换外，还需要在前向传播的注意力激活值 $Z$ 后插入一个在线的“Hadamard heads”计算块（计算 $Z(H_{n_h} \otimes I)$），从而高效完成了 Value 空间的完全旋转 。



#### Stage 1d: Key 向量旋转与 Caching 策略 (Key Rotation)

注意力机制中的 Key 向量同样饱受离群值困扰，但由于位置编码（RoPE）的存在，无法像 Value 一样直接将哈达玛矩阵融合进 $W_q$ 和 $W_k$ 中（RoPE 会破坏旋转后的线性关系） 。

* **在线 Head-wise 旋转：** 作者选择在 RoPE 操作**之后**，利用在线计算的方式对 Query 和 Key 向量直接进行头内哈达玛旋转 ：

$$Q \leftarrow \text{Pos}(XW_q)(I \otimes H_{d_h}), \quad K \leftarrow \text{Pos}(XW_k)(I \otimes H_{d_h})$$


由于 $Q$ 和 $K$ 被施加了相同的正交变换，在计算点积注意力得分 $QK^{\top}$ 时，旋转效应在数学上会自动抵消，保证了 Softmax 得分不发生任何改变 。


* **Post-RoPE Caching（后置位置编码缓存）：** 在 Caching 策略的选择上，相较于 Pre-RoPE Caching（其需要在每次解码时对所有历史 KV 重新进行逆旋转计算，开销极大） ，QuaRot 采用了 **Post-RoPE Caching** 策略 。这意味着在 Decoding（解码）阶段，**每个时间步只需要对当前生成的单一个 Token 运行一次哈达玛变换**，极大地加速了推理速度 。

### 阶段 2：低比特量化实施 (Stage 2: Quantization Operations)

在通过第一阶段将全网流通的信号全部“抹平”且去除离群值后，第二阶段正式引入低比特量化算子 。

#### Stage 2a: 权重物理量化 (Weight Quantization)

* **默认配置：** 方案默认使用 **GPTQ** 算法在全精度修改后的模型上进行离线权重质变量化，将其压入 INT4 。


* **灵活性：** 得益于不相干处理的优异性质，QuaRot 展现出极高的量化算法兼容性 。即使放弃复杂的 GPTQ，直接使用开销极低、无需校准集的四舍五入到最近整数（RTN）方案，在模型规模较大时（如 70B）也能够保持极低的准确率损失 。

#### Stage 2b: 激活值在线量化 (Online Quantization Operations)

* **混合精度保留：** 将不含缩放参数的纯矩阵范数计算（RMSNorm）保留在 FP32 中运行，以稳定基础流 。
* **对称每 Token 量化（Symmetric Per-token Quantization）：** 在进入各个线性层前，对激活值矩阵进行逐行（Row/Token）的对称量化 。动态缩放因子（Row Scales）的计算方法为：提取该行绝对值的最大值并除以 7（即 INT4 能表示的最大正整数） 。
* **反量化机制：** 在 TensorCore 完成高效的低精度矩阵乘法（输入 INT4 $\times$ 权重 INT4 $\rightarrow$ 累加器 INT32）后，将结果铸造型（Cast）回 FP16 精度，并乘上对应的输入行缩放因子与权重列缩放因子，无缝恢复至模型默认的半精度空间 。

#### Stage 2c: 量化注意力层与 KV Cache (Quantized Attention)

长文本和大批次推理的核心瓶颈在于显存（I/O 绑定） 。由于 Stage 1d 已经将 Key 和 Value 的离群值完全消除，此处可以安全地对整个 KV Cache 进行低比特量化 。

* **精细化控制：** 保持 Query 向量为 FP16 不向量化 。
* **在线反量化流：** 采用类似于 Flash Attention 的在线 Softmax 计算机制 。当硬件从显存中加载分段的（Grouped）量化 KV 向量到 SRAM 时，在计算单元内部实时进行反量化，并在 FP16 下计算点积 。这大幅减少了慢速显存（VRAM）与快速缓存（SRAM）之间的 **I/O 数据传输量**，从而直接打破了大模型解码阶段的带宽瓶颈 。

---

## 项目代码

fake_quant/ 目录是核心代码，我们从 fake_quant/main.py 入手，它的核心逻辑严格对应了论文的四个主要步骤：

### Stage 1: Rotation

```python
# Rotate the weights
if args.rotate:
    rotation_utils.fuse_layer_norms(model)  # Step 1: 融合归一化层
    rotation_utils.rotate_model(model, args) # Step 2: 全局正交矩阵旋转权重
    utils.cleanup_memory(verbos=True)
        
    quant_utils.add_actquant(model) # Step 3: 为所有线性层包裹一层激活值量化外壳
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if 'down_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True # 标记 FFN 内部需要运行在线哈达玛
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had
        if 'o_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True # 标记 Attention 输出侧需要运行跨头哈达玛
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            qlayers[name].fp32_had = args.fp32_had
else:
    quant_utils.add_actquant(model) # 如果不旋转，也要加外壳（作为 Baseline 对比）
```

#### Stage 1a. 模型旋转

离线将 RMSNorm 融合并旋转权重

* `fuse_layer_norms`：将 `RMSNorm` 的归一化缩放因子直接乘进后面的权重矩阵，使其退化为纯范数计算，为旋转腾出数学空间 。
* `rotate_model`：生成全局随机哈达玛矩阵 $Q$，直接修改模型各线性层的权重，使层间流通的激活值全部进入旋转空间，**从而干掉层间激活值的离群值** 。
* `add_actquant`：为模型的线性层动态套上一个 Wrapper（包装层）。这个包装层不仅负责量化，还负责**执行在线哈达玛变换** 。

#### Stage 1b/1c. 配置在线哈达玛变换

标记哪些层（如 down_proj, o_proj）在前向传播中需要运行在线哈达玛旋转

* **针对 `down_proj` 的配置（Stage 1b）：** FFN 模块的内部激活值会产生新离群值 。代码将其 `online_full_had` 设为 `True`，这意味着在前向传播时，进该层矩阵乘法前，激活值会强行通过 `hadamard_utils` 运行一次完整的在线快速哈达玛变换（WHT） 。
* **针对 `o_proj` 的配置（Stage 1c）：** 对应自注意力模块输出投影，开启 `online_partial_had` 运行跨头（Head-wise）的哈达玛共享变换 。
 
### Stage 2: Quantization

#### Stage 2a. 权重低比特量化： 使用 GPTQ 或 RTN

```python
if args.w_bits < 16:
	save_dict = {}
	# 路径 A：直接加载已经量化好的旋转权重
	if args.load_qmodel_path: # Load Quantized Rotated Model
		assert args.rotate, "Model should be rotated to load a quantized model!"
		assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
		print("Load quantized model from ", args.load_qmodel_path)
		save_dict = torch.load(args.load_qmodel_path)
		model.load_state_dict(save_dict["model"])
	
	# 路径 B：调用校准集，运行 GPTQ 算法进行 4-bit 量化
	elif not args.w_rtn: # GPTQ Weight Quantization
		assert "llama" in args.model, "Only llama is supported for GPTQ!"
		
		trainloader = data_utils.get_loaders(
			args.cal_dataset, nsamples=args.nsamples,
			seed=args.seed, model=args.model,
			seqlen=model.seqlen, eval_mode=False
		)
		quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
		save_dict["w_quantizers"] = quantizers
	
	# 路径 C：无需校准集，直接暴力进行 RTN (Round-to-Nearest) 量化
	else: # RTN Weight Quantization
		quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
		save_dict["w_quantizers"] = quantizers
```

#### Stage 2b/2c. 配置激活值与 KV Cache 的在线量化

为输入流、Value 和 Key 动态注入伪量化器

```python
# Add Input Quantization
if args.a_bits < 16 or args.v_bits < 16:
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    down_proj_groupsize = -1
    if args.a_groupsize > 0 and "llama" in args.model:
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
    
    for name in qlayers:            
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize
        layer_a_sym = not(args.a_asym)
        layer_a_clip = args.a_clip_ratio
        
        if 'v_proj' in name and args.v_bits < 16: # 配置自注意力的 Value 缓存量化精度
            qlayers[name].out_quantizer.configure(bits=args.v_bits, groupsize=args.v_groupsize,
                                          sym=not(args.v_asym), clip_ratio=args.v_clip_ratio)
        
        if 'lm_head' in name: # 约束：lm_head 通常不进行低比特量化，保持高精度 
            layer_input_bits = 16
        
        if 'down_proj' in name: # 特殊配置：由于 down_proj 特别敏感，有时强制开启 INT8 [cite: 75]
            if args.int8_down_proj:
                layer_input_bits = 8
            layer_groupsize = down_proj_groupsize

        # 将配置正式写入该层的量化器中
        qlayers[name].quantizer.configure(bits=layer_input_bits, groupsize=layer_groupsize,
                                          sym=layer_a_sym, clip_ratio=layer_a_clip)

# 针对 Key 缓存的量化配置 (Stage 1d)
if args.k_bits < 16:
    if args.k_pre_rope:
        raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
    else:
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                      "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
        for layer in layers:
            # 核心注入：在 RoPE 位置编码函数执行完后，立即切入动态哈达玛旋转与 Key 量化 [cite: 431]
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                        layer.self_attn, rope_function_name, config=model.config, **k_quant_config)
```

* `qlayers[name].quantizer.configure(...)`：对网络流通的每一层 **Activation（输入激活值）** 注入量化参数（如指定比特数、对称/非对称、张量裁剪率等） 。
* `v_proj` 处的 `out_quantizer`：专门针对自注意力层输出的 **Value 向量** 缓存配置量化参数 。
* `add_qk_rotation_wrapper_after_function_call_in_forward`：**专门处理 Key 向量**。由于位置编码（RoPE）会破坏离线融合的数学连续性，代码通过钩子在 `Forward` 流程中寻找 `apply_rotary_pos_emb` 这一 RoPE 函数，在其**执行完的刹那间**，在线对 Key 向量执行哈达玛旋转并实施低比特量化 。这对应了论文中提出的 **Post-RoPE Caching（后置位置编码缓存）** 策略 。

### 端到端评估

测定 WikiText-2 困惑度（PPL）和 Zero-shot 任务准确率

