# Triton Ops in vLLM

## Env

Common Infos:

```
torch                             2.3.1
torch-npu                         2.3.1.post2
triton-ascend                     36969dafdad51877233f6adb2c077212d5058f1d
```

Prepare vLLM repo:

```bash
git clone https://github.com/shink/vllm.git -b jyh/triton
```

### vllm/attention/ops/chunked_prefill_paged_decode.py

- [x] `cdiv_fn`
- [x] `kernel_paged_attention_2d`

```python
pytest -svx tests/kernels/attention/test_prefix_prefill.py -k "chunked_prefill_paged_decode"
```

<details>
<summary>MLIRCompilationError</summary>

```
E               triton.compiler.errors.MLIRCompilationError:
E               ///------------------[ERROR][Triton][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: remark: [MaskState] Unsupported cmpi scenario
E                       qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: note: see current operation: %142 = arith.cmpi sge, %107, %141 : tensor<128x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: remark: [MaskState] Unsupported cmpi scenario
E                       qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: note: see current operation: %131 = arith.cmpi sge, %98, %130 : tensor<128x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:156:26: error: cannot div 0!
E                               K_cache + off_k,
E                                        ^
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp148mb23j/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp148mb23j/kernel.ttadapter.mlir
E                #0 0x0000aaaacdd0ef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
E                #1 0x0000aaaacdd0ca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
E                #2 0x0000aaaacdd0cb78 SignalHandler(int) Signals.cpp:0:0
E                #3 0x0000ffffb7d3e7c0 (linux-vdso.so.1+0x7c0)
E                #4 0x0000aaaacd0457e0 mlir::mulOpFoldResult(mlir::OpFoldResult const&, mlir::Value const&, mlir::Location const&, mlir::OpBuilder&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x2e57e0)
E                #5 0x0000aaaaccecc3a4 mlir::triton::BlockData::mulBlock(mlir::triton::BlockData&, mlir::triton::BlockData&, mlir::Location, mlir::ConversionPatternRewriter&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c3a4)
E                #6 0x0000aaaaccecf254 mlir::triton::BlockDataParser::parseMul(mlir::arith::MulIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f254)
E                #7 0x0000aaaaccece74c mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e74c)
E                #8 0x0000aaaaccece9dc mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e9dc)
E                #9 0x0000aaaaccecf06c mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f06c)
E               #10 0x0000aaaaccece764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
E               #11 0x0000aaaacced0160 mlir::triton::BlockDataParser::parseBroadcast(mlir::triton::BroadcastOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x170160)
E               #12 0x0000aaaaccece988 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e988)
E               #13 0x0000aaaaccecf040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
E               #14 0x0000aaaaccece764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
E               #15 0x0000aaaaccecf040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
E               #16 0x0000aaaaccece764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
E               #17 0x0000aaaaccecdeb0 mlir::triton::BlockDataParser::parseAddPtr(mlir::triton::AddPtrOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16deb0)
E               #18 0x0000aaaacced34fc mlir::triton::BlockDataParser::rewriteAddPtr(mlir::triton::AddPtrOp, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> >&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x1734fc)
E               #19 0x0000aaaacceb4ed0 LoadStoreConverter::AddPtrConverter::matchAndRewrite(mlir::triton::AddPtrOp, mlir::triton::AddPtrOpAdaptor, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x154ed0)
E               #20 0x0000aaaaccea0cb0 mlir::OpConversionPattern<mlir::triton::AddPtrOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x140cb0)
E               #21 0x0000aaaacd96eeec mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc0eeec)
E               #22 0x0000aaaacd995ab0 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc35ab0)
E               #23 0x0000aaaacd972d24 (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*, mlir::ConversionPatternRewriter&) DialectConversion.cpp:0:0
E               #24 0x0000aaaacd9731a8 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc131a8)
E               #25 0x0000aaaacd978788 mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc18788)
E               #26 0x0000aaaacd9794f4 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc194f4)
E               #27 0x0000aaaacce97848 (anonymous namespace)::TritonToLinalgPass::runOnOperation() TritonToLinalgPass.cpp:0:0
E               #28 0x0000aaaacd934974 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4974)
E               #29 0x0000aaaacd934df0 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4df0)
E               #30 0x0000aaaacd935cb8 mlir::PassManager::run(mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd5cb8)
E               #31 0x0000aaaacd929a38 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
E               #32 0x0000aaaacd92a0bc processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
E               #33 0x0000aaaacd92a1f8 llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) MlirOptMain.cpp:0:0
E               #34 0x0000aaaacdca1328 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf41328)
E               #35 0x0000aaaacd9240bc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc40bc)
E               #36 0x0000aaaacd92a308 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca308)
E               #37 0x0000aaaacd92a6fc mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca6fc)
E               #38 0x0000aaaacce6e548 main (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x10e548)
E               #39 0x0000ffffb77b73fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
E               #40 0x0000ffffb77b74cc call_init ./csu/../csu/libc-start.c:128:20
E               #41 0x0000ffffb77b74cc __libc_start_main ./csu/../csu/libc-start.c:379:5
E               #42 0x0000aaaacce8de70 _start (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x12de70)
E               ///------------------[ERROR][Triton][END]------------------

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:297: MLIRCompilationError
```

</details>

### vllm/attention/ops/prefix_prefill.py

- [x] `_fwd_kernel` —— 会调用 `tl.multiple_of`
- [ ] `_fwd_kernel_flash_attn_v2`
- [x] `_fwd_kernel_alibi`

```python
pytest -svx tests/kernels/attention/test_prefix_prefill.py -k "context_attention_fwd"
```

<details>
<summary>MLIRCompilationError</summary>

```
E               triton.compiler.errors.MLIRCompilationError:
E               ///------------------[ERROR][Triton][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: remark: [MaskState] Unsupported cmpi scenario
E                       qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: note: see current operation: %141 = arith.cmpi sge, %106, %140 : tensor<128x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: remark: [MaskState] Unsupported cmpi scenario
E                       qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: note: see current operation: %130 = arith.cmpi sge, %97, %129 : tensor<128x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:156:26: error: cannot div 0!
E                               K_cache + off_k,
E                                        ^
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmpujdhfv1c/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpujdhfv1c/kernel.ttadapter.mlir
E                #0 0x0000aaaae361ef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
E                #1 0x0000aaaae361ca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
E                #2 0x0000aaaae361cb78 SignalHandler(int) Signals.cpp:0:0
E                #3 0x0000ffff9527c7c0 (linux-vdso.so.1+0x7c0)
E                #4 0x0000aaaae29557e0 mlir::mulOpFoldResult(mlir::OpFoldResult const&, mlir::Value const&, mlir::Location const&, mlir::OpBuilder&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x2e57e0)
E                #5 0x0000aaaae27dc3a4 mlir::triton::BlockData::mulBlock(mlir::triton::BlockData&, mlir::triton::BlockData&, mlir::Location, mlir::ConversionPatternRewriter&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c3a4)
E                #6 0x0000aaaae27df254 mlir::triton::BlockDataParser::parseMul(mlir::arith::MulIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f254)
E                #7 0x0000aaaae27de74c mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e74c)
E                #8 0x0000aaaae27de9dc mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e9dc)
E                #9 0x0000aaaae27df06c mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f06c)
E               #10 0x0000aaaae27de764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
E               #11 0x0000aaaae27e0160 mlir::triton::BlockDataParser::parseBroadcast(mlir::triton::BroadcastOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x170160)
E               #12 0x0000aaaae27de988 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e988)
E               #13 0x0000aaaae27df040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
E               #14 0x0000aaaae27de764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
E               #15 0x0000aaaae27df040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
E               #16 0x0000aaaae27de764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
E               #17 0x0000aaaae27ddeb0 mlir::triton::BlockDataParser::parseAddPtr(mlir::triton::AddPtrOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16deb0)
E               #18 0x0000aaaae27e34fc mlir::triton::BlockDataParser::rewriteAddPtr(mlir::triton::AddPtrOp, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> >&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x1734fc)
E               #19 0x0000aaaae27c4ed0 LoadStoreConverter::AddPtrConverter::matchAndRewrite(mlir::triton::AddPtrOp, mlir::triton::AddPtrOpAdaptor, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x154ed0)
E               #20 0x0000aaaae27b0cb0 mlir::OpConversionPattern<mlir::triton::AddPtrOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x140cb0)
E               #21 0x0000aaaae327eeec mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc0eeec)
E               #22 0x0000aaaae32a5ab0 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc35ab0)
E               #23 0x0000aaaae3282d24 (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*, mlir::ConversionPatternRewriter&) DialectConversion.cpp:0:0
E               #24 0x0000aaaae32831a8 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc131a8)
E               #25 0x0000aaaae3288788 mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc18788)
E               #26 0x0000aaaae32894f4 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc194f4)
E               #27 0x0000aaaae27a7848 (anonymous namespace)::TritonToLinalgPass::runOnOperation() TritonToLinalgPass.cpp:0:0
E               #28 0x0000aaaae3244974 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4974)
E               #29 0x0000aaaae3244df0 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4df0)
E               #30 0x0000aaaae3245cb8 mlir::PassManager::run(mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd5cb8)
E               #31 0x0000aaaae3239a38 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
E               #32 0x0000aaaae323a0bc processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
E               #33 0x0000aaaae323a1f8 llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) MlirOptMain.cpp:0:0
E               #34 0x0000aaaae35b1328 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf41328)
E               #35 0x0000aaaae32340bc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc40bc)
E               #36 0x0000aaaae323a308 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca308)
E               #37 0x0000aaaae323a6fc mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca6fc)
E               #38 0x0000aaaae277e548 main (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x10e548)
E               #39 0x0000ffff94ce73fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
E               #40 0x0000ffff94ce74cc call_init ./csu/../csu/libc-start.c:128:20
E               #41 0x0000ffff94ce74cc __libc_start_main ./csu/../csu/libc-start.c:379:5
E               #42 0x0000aaaae279de70 _start (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x12de70)
E               ///------------------[ERROR][Triton][END]------------------

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:297: MLIRCompilationError
```

</details>

### vllm/attention/ops/triton_decode_attention.py

- [x] `tanh`
- [x] `_fwd_kernel_stage1`
- [x] `_fwd_grouped_kernel_stage1`
- [x] `_fwd_kernel_stage2`

```python
pytest -svx tests/kernels/attention/test_triton_decode_attention.py
```

> 问题跟踪：https://gitee.com/ascend/triton-ascend/issues/IC5HR1?from=project-issue

<details>
<summary>MLIRCompilationError</summary>

```
E               triton.compiler.errors.MLIRCompilationError:
E               ///------------------[ERROR][Triton][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp7qif0584/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp7qif0584/kernel.ttadapter.mlir
E               #0 0x0000aaaab15eef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
E               #1 0x0000aaaab15eca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
E               #2 0x0000aaaab15ecb78 SignalHandler(int) Signals.cpp:0:0
E               #3 0x0000ffffaea4d7c0 (linux-vdso.so.1+0x7c0)
E               #4 0x0000ffffae537d54 ./string/../sysdeps/aarch64/multiarch/../memcpy.S:257:0
E               #5 0x0000aaaab07af6f8 mlir::triton::BlockDataParser::parseExpandDims(mlir::triton::ExpandDimsOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f6f8)
E               #6 0x0000fffffbece5f0
E               ///------------------[ERROR][Triton][END]------------------

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:297: MLIRCompilationError
```

</details>

### vllm/model_executor/layers/lightning_attn.py

- [x] `_fwd_diag_kernel`
- [x] `_fwd_kv_parallel`
- [x] `_fwd_kv_reduce`
- [x] `_fwd_none_diag_kernel`
- [x] `_linear_attn_decode_kernel`

```python
pytest -svx tests/kernels/attention/test_lightning_attn.py
```

> 缺少 return 操作数，见：https://mlir.llvm.org/docs/Dialects/Func/#funcreturn-funcreturnop

<details>
<summary>MLIRCompilationError</summary>

```
E               triton.compiler.errors.MLIRCompilationError:
E               ///------------------[ERROR][Triton][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/model_executor/layers/lightning_attn.py:507:0: error: 'func.return' op has 1 operands, but enclosing function (@_linear_attn_decode_kernel) returns 0
E               /home/devuser/workspace/vllm-project/vllm/vllm/model_executor/layers/lightning_attn.py:507:0: note: see current operation: "func.return"(%9) : (i1) -> ()
E               ///------------------[ERROR][Triton][END]------------------

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:297: MLIRCompilationError
```

</details>

### vllm/model_executor/layers/mamba/ops/ssd_state_passing.py

- [x] `_state_passing_fwd_kernel`

```python
python triton/state_passing_fwd/test.py
```

<details>
<summary>MLIRCompilationError</summary>

```
Traceback (most recent call last):
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/compiler/compiler.py", line 289, in compile
    next_module = compile_ir(module, metadata)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 311, in <lambda>
    stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile(src, metadata, options)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 181, in linalg_to_bin_enable_npu_compile
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/home/devuser/.conda/envs/triton/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/devuser/program/bisheng/compiler/npuc', '/tmp/tmpi1uwsy4o/kernel.ttadapter.mlir', '--enable-auto-multi-buffer=True', '-o', '/tmp/tmpi1uwsy4o/kernel']' died with <Signals.SIGABRT: 6>.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/devuser/workspace/vllm-project/vllm/triton/state_passing_fwd/test.py", line 34, in <module>
    fn()
  File "/home/devuser/workspace/vllm-project/vllm/triton/state_passing_fwd/test.py", line 29, in fn
    ssd_state_passing._state_passing_fwd(states, dA_chunk_cumsum)
  File "/home/devuser/workspace/vllm-project/vllm/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py", line 178, in _state_passing_fwd
    _state_passing_fwd_kernel[grid](
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/autotuner.py", line 194, in run
    ret = self.fn.run(
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/compiler/compiler.py", line 297, in compile
    raise MLIRCompilationError(stage_name, e.stderr.decode('utf-8'))
triton.compiler.errors.MLIRCompilationError:
///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
error: strides must not be zero
npuc: /home/JenkinsStub/llvm-project/mlir/include/mlir/IR/StorageUniquerSupport.h:181: static ConcreteT mlir::detail::StorageUserBase<mlir::StridedLayoutAttr, mlir::Attribute, mlir::detail::StridedLayoutAttrStorage, mlir::detail::AttributeUniquer, Trait>::get(mlir::MLIRContext *, Args &&...) [ConcreteT = mlir::StridedLayoutAttr, BaseT = mlir::Attribute, StorageT = mlir::detail::StridedLayoutAttrStorage, UniquerT = mlir::detail::AttributeUniquer, Traits = <Trait>, Args = <long, llvm::ArrayRef<long>>]: Assertion `succeeded(ConcreteT::verify(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/program/bisheng/compiler/npuc /tmp/tmpi1uwsy4o/kernel.ttadapter.mlir --enable-auto-multi-buffer=True -o /tmp/tmpi1uwsy4o/kernel
 #0 0x0000000004295f18 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/program/bisheng/compiler/npuc+0x4295f18)
 #1 0x0000000004293c30 llvm::sys::RunSignalHandlers() (/home/devuser/program/bisheng/compiler/npuc+0x4293c30)
 #2 0x0000000004296660 SignalHandler(int) Signals.cpp:0:0
 #3 0x0000ffff869847c0 (linux-vdso.so.1+0x7c0)
 #4 0x0000ffff863bf1f0 __pthread_kill_implementation ./nptl/./nptl/pthread_kill.c:44:76
 #5 0x0000ffff8637a67c gsignal ./signal/../sysdeps/posix/raise.c:27:6
 #6 0x0000ffff86367130 abort ./stdlib/./stdlib/abort.c:81:7
 #7 0x0000ffff86373fd4 __assert_fail_base ./assert/./assert/assert.c:91:7
 #8 0x0000ffff8637404c (/lib/aarch64-linux-gnu/libc.so.6+0x3404c)
 #9 0x0000000003e83840 mlir::StridedLayoutAttr::getOffset() const (/home/devuser/program/bisheng/compiler/npuc+0x3e83840)
#10 0x0000000003e83768 mlir::StridedLayoutAttr::get(mlir::MLIRContext*, long, llvm::ArrayRef<long>) (/home/devuser/program/bisheng/compiler/npuc+0x3e83768)
#11 0x0000000003795784 ReinterpretCastReturnTypeCanonicalizer::operator()(mlir::memref::ReinterpretCastOp, llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<mlir::OpFoldResult>) MemRefOps.cpp:0:0
#12 0x0000000003795500 mlir::OpWithOffsetSizesAndStridesConstantArgumentFolder<mlir::memref::ReinterpretCastOp, ReinterpretCastReturnTypeCanonicalizer, ReinterpretCastCanonicalizer>::matchAndRewrite(mlir::memref::ReinterpretCastOp, mlir::PatternRewriter&) const MemRefOps.cpp:0:0
#13 0x00000000037949d4 mlir::detail::OpOrInterfaceRewritePatternBase<mlir::memref::ReinterpretCastOp>::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const MemRefOps.cpp:0:0
#14 0x0000000003a2a538 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<mlir::LogicalResult (mlir::Pattern const&)>)::$_2::operator()() const PatternApplicator.cpp:0:0
#15 0x0000000003a275dc mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<mlir::LogicalResult (mlir::Pattern const&)>) (/home/devuser/program/bisheng/compiler/npuc+0x3a275dc)
#16 0x0000000003a1597c (anonymous namespace)::GreedyPatternRewriteDriver::processWorklist() GreedyPatternRewriteDriver.cpp:0:0
#17 0x0000000003a120d8 mlir::applyPatternsAndFoldGreedily(mlir::Region&, mlir::FrozenRewritePatternSet const&, mlir::GreedyRewriteConfig, bool*) (/home/devuser/program/bisheng/compiler/npuc+0x3a120d8)
#18 0x00000000039c3758 (anonymous namespace)::Canonicalizer::runOnOperation() Canonicalizer.cpp:0:0
#19 0x0000000003a6d888 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/program/bisheng/compiler/npuc+0x3a6d888)
#20 0x0000000003a6dfe8 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/program/bisheng/compiler/npuc+0x3a6dfe8)
#21 0x0000000003a70028 mlir::PassManager::run(mlir::Operation*) (/home/devuser/program/bisheng/compiler/npuc+0x3a70028)
#22 0x00000000016b391c bishengir::runPipeline(mlir::ModuleOp, std::function<void (mlir::PassManager&, bishengir::BiShengPipelineOptions const&)> const&, bishengir::BiShengPipelineOptions const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) (/home/devuser/program/bisheng/compiler/npuc+0x16b391c)
#23 0x00000000016bd944 bishengir::runBiShengPipeline(mlir::ModuleOp, bishengir::BiShengPipelineOptions const&) (/home/devuser/program/bisheng/compiler/npuc+0x16bd944)
#24 0x0000000001649ab0 main (/home/devuser/program/bisheng/compiler/npuc+0x1649ab0)
#25 0x0000ffff863673fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#26 0x0000ffff863674cc call_init ./csu/../csu/libc-start.c:128:20
#27 0x0000ffff863674cc __libc_start_main ./csu/../csu/libc-start.c:379:5
#28 0x0000000001647c6c _start (/home/devuser/program/bisheng/compiler/npuc+0x1647c6c)
///------------------[ERROR][Triton][END]------------------

[ERROR] 2025-04-30-07:44:05 (PID:3754768, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```

</details>

### vllm/model_executor/layers/quantization/awq_triton.py

- [x] `awq_dequantize_kernel`
- [x] `awq_gemm_kernel`

```python
pytest -svx tests/kernels/quantization/test_awq_triton.py
```

<details>
<summary>MLIRCompilationError</summary>

```
E               triton.compiler.errors.MLIRCompilationError:
E               ///------------------[ERROR][Triton][BEG]------------------
E               [ConvertLinalgRToBinary] encounters error:
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmptwao82mw/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               Error run BiShengIR pipeline pipeline
E               ///------------------[ERROR][Triton][END]------------------

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:297: MLIRCompilationError
```

</details>

### vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py

- [x] `scaled_mm_kernel`

```python
pytest -svx tests/kernels/quantization/test_triton_scaled_mm.py
```

> 问题跟踪：https://gitee.com/ascend/triton-ascend/issues/IC5IDP?from=project-issue

<details>
<summary>MLIRCompilationError</summary>

```
E               triton.compiler.errors.MLIRCompilationError:
E               ///------------------[ERROR][Triton][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmpp5k26p28/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpp5k26p28/kernel.ttadapter.mlir
E                #0 0x0000aaaab457ef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
E                #1 0x0000aaaab457ca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
E                #2 0x0000aaaab457cb78 SignalHandler(int) Signals.cpp:0:0
E                #3 0x0000ffffbebe57c0 (linux-vdso.so.1+0x7c0)
E                #4 0x0000aaaab44f8558 mlir::Value::getDefiningOp() const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf28558)
E                #5 0x0000aaaab3740d68 mlir::triton::BlockDataParser::parseUnrealizedCast(mlir::UnrealizedConversionCastOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x170d68)
E                #6 0x0000aaaab37462f4 mlir::triton::BlockDataParser::rewriteForOp(mlir::scf::ForOp, mlir::ConversionPatternRewriter&, std::map<int, std::set<int, std::less<int>, std::allocator<int> >, std::less<int>, std::allocator<std::pair<int const, std::set<int, std::less<int>, std::allocator<int> > > > >&, int, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> >&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x1762f4)
E                #7 0x0000aaaab37307cc TTOpConverters::LoopConverter::matchAndRewrite(mlir::scf::ForOp, mlir::scf::ForOpAdaptor, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x1607cc)
E                #8 0x0000aaaab3711040 mlir::OpConversionPattern<mlir::scf::ForOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x141040)
E                #9 0x0000aaaab41deeec mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc0eeec)
E               #10 0x0000aaaab4205ab0 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc35ab0)
E               #11 0x0000aaaab41e2d24 (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*, mlir::ConversionPatternRewriter&) DialectConversion.cpp:0:0
E               #12 0x0000aaaab41e31a8 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc131a8)
E               #13 0x0000aaaab41e8788 mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc18788)
E               #14 0x0000aaaab41e94f4 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc194f4)
E               #15 0x0000aaaab3707848 (anonymous namespace)::TritonToLinalgPass::runOnOperation() TritonToLinalgPass.cpp:0:0
E               #16 0x0000aaaab41a4974 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4974)
E               #17 0x0000aaaab41a4df0 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4df0)
E               #18 0x0000aaaab41a5cb8 mlir::PassManager::run(mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd5cb8)
E               #19 0x0000aaaab4199a38 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
E               #20 0x0000aaaab419a0bc processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
E               #21 0x0000aaaab419a1f8 llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) MlirOptMain.cpp:0:0
E               #22 0x0000aaaab4511328 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf41328)
E               #23 0x0000aaaab41940bc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc40bc)
E               #24 0x0000aaaab419a308 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca308)
E               #25 0x0000aaaab419a6fc mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca6fc)
E               #26 0x0000aaaab36de548 main (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x10e548)
E               #27 0x0000ffffbe6573fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
E               #28 0x0000ffffbe6574cc call_init ./csu/../csu/libc-start.c:128:20
E               #29 0x0000ffffbe6574cc __libc_start_main ./csu/../csu/libc-start.c:379:5
E               #30 0x0000aaaab36fde70 _start (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x12de70)
E               ///------------------[ERROR][Triton][END]------------------

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:297: MLIRCompilationError
```

</details>

### vllm/v1/sample/rejection_sampler.py

- [x] `rejection_greedy_sample_kernel`
- [x] `rejection_random_sample_kernel`
- [x] `expand_kernel`
- [x] `sample_recovered_tokens_kernel`

```python
pytest -svx tests/v1/sample/test_rejection_sampler.py -k test_deterministic_when_seeded
```

<details>
<summary>MLIRCompilationError</summary>

```
================================================================================================================== FAILURES ==================================================================================================================
______________________________________________________________________________________________ test_deterministic_when_seeded[20-0.0-1-1000-1] _______________________________________________________________________________________________

src = <triton.compiler.compiler.ASTSource object at 0xffffb3fe1000>, target = GPUTarget(backend='npu', arch='Ascend910B3', warp_size=0)
options = NPUOptions(debug=False, sanitize_overflow=True, llvm_version=15, kernel_name='triton_', cluster_dims=(1, 1, 1), num_wa...nput_precision='ieee', enable_npu_compile=True, max_num_imprecise_acc_default=None, extern_libs=None, multibuffer=True)

    def compile(src, target=None, options=None):
        if target is None:
            target = driver.active.get_current_target()
        assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
        backend = make_backend(target)
        ir_source = not isinstance(src, ASTSource)
        # create backend
        if ir_source:
            assert isinstance(src, str), "source must be either AST or a filepath"
            src = IRSource(src)
        extra_options = src.parse_options()
        options = backend.parse_options(dict(options or dict(), **extra_options))
        # create cache manager
        env_vars = get_cache_invalidating_env_vars()
        key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
        hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        fn_cache_manager = get_cache_manager(hash)
        # For dumping/overriding only hash the source as we want it to be independent of triton
        # core changes to make it easier to track kernels by hash.
        enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
        enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
        fn_override_manager = get_override_manager(src.hash()) if enable_override else None
        fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
        # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
        # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
        # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
        # the file name to 150 characters to be safe.
        file_name = src.name[:150]
        metadata_filename = f"{file_name}.json"
        metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
        metadata_path = metadata_group.get(metadata_filename)
        always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0") == "1"
        if not always_compile and metadata_path is not None:
            # cache hit!
            metadata = json.loads(Path(metadata_path).read_text())
            return CompiledKernel(src, metadata_group, hash)
        compile_speed_opt = os.getenv("TRITON_ASCEND_COMPILE_SPEED_OPT", 'false').lower() in ('true', '1')
        if (compile_speed_opt):
            ttir_path = f"{file_name}.ttir"
            if (metadata_path is None) and (fn_cache_manager.has_file(ttir_path)):
                # Already compile once but failed. So directly return
                raise Exception("already failed once")
        # initialize metadata
        metadata = {
            "hash": hash,
            "target": target,
            **options.__dict__,
            **env_vars,
        }
        # run compilation pipeline  and populate metadata
        stages = dict()
        backend.add_stages(stages, options)
        first_stage = list(stages.keys()).index(src.ext)
        # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
        if ir_source:
            first_stage += 1
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation()
        module_map = backend.get_module_map()
        try:
            module = src.make_ir(options, codegen_fns, module_map, context)
        except Exception as e:
            filter_traceback(e)
            raise
        use_ir_loc = os.environ.get("USE_IR_LOC", None)
        for ext, compile_ir in list(stages.items())[first_stage:]:
            try:
>               next_module = compile_ir(module, metadata)

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:289:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
../../ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py:310: in <lambda>
    stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
../../ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py:60: in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

input = None, capture_output = True, timeout = None, check = True
popenargs = (['/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmpwbl....ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmpwbl0_tcv/kernel.ttadapter.mlir'],)
kwargs = {'stderr': -1, 'stdout': -1}, process = <Popen: returncode: 1 args: ['/home/devuser/workspace/ascend/triton-ascend/t...>, stdout = b''
stderr = b'/home/devuser/workspace/vllm-project/vllm/vllm/v1/sample/rejection_sampler.py:569:0: error: \'func.return\' op has 1...project/vllm/vllm/v1/sample/rejection_sampler.py:569:0: note: see current operation: "func.return"(%12) : (i1) -> ()\n'
retcode = 1

    def run(*popenargs,
            input=None, capture_output=False, timeout=None, check=False, **kwargs):
        """Run command with arguments and return a CompletedProcess instance.

        The returned instance will have attributes args, returncode, stdout and
        stderr. By default, stdout and stderr are not captured, and those attributes
        will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
        or pass capture_output=True to capture both.

        If check is True and the exit code was non-zero, it raises a
        CalledProcessError. The CalledProcessError object will have the return code
        in the returncode attribute, and output & stderr attributes if those streams
        were captured.

        If timeout is given, and the process takes too long, a TimeoutExpired
        exception will be raised.

        There is an optional argument "input", allowing you to
        pass bytes or a string to the subprocess's stdin.  If you use this argument
        you may not also use the Popen constructor's "stdin" argument, as
        it will be used internally.

        By default, all communication is in bytes, and therefore any "input" should
        be bytes, and the stdout and stderr will be bytes. If in text mode, any
        "input" should be a string, and stdout and stderr will be strings decoded
        according to locale encoding, or by "encoding" if set. Text mode is
        triggered by setting any of text, encoding, errors or universal_newlines.

        The other arguments are the same as for the Popen constructor.
        """
        if input is not None:
            if kwargs.get('stdin') is not None:
                raise ValueError('stdin and input arguments may not both be used.')
            kwargs['stdin'] = PIPE

        if capture_output:
            if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
                raise ValueError('stdout and stderr arguments may not be used '
                                 'with capture_output.')
            kwargs['stdout'] = PIPE
            kwargs['stderr'] = PIPE

        with Popen(*popenargs, **kwargs) as process:
            try:
                stdout, stderr = process.communicate(input, timeout=timeout)
            except TimeoutExpired as exc:
                process.kill()
                if _mswindows:
                    # Windows accumulates the output in a single blocking
                    # read() call run on child threads, with the timeout
                    # being done in a join() on those threads.  communicate()
                    # _after_ kill() is required to collect that and add it
                    # to the exception.
                    exc.stdout, exc.stderr = process.communicate()
                else:
                    # POSIX _communicate already populated the output so
                    # far into the TimeoutExpired exception.
                    process.wait()
                raise
            except:  # Including KeyboardInterrupt, communicate handled that.
                process.kill()
                # We don't call process.wait() as .__exit__ does that for us.
                raise
            retcode = process.poll()
            if check and retcode:
>               raise CalledProcessError(retcode, process.args,
                                         output=stdout, stderr=stderr)
E               subprocess.CalledProcessError: Command '['/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmpwbl0_tcv/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmpwbl0_tcv/kernel.ttadapter.mlir']' returned non-zero exit status 1.

../../../.conda/envs/triton/lib/python3.10/subprocess.py:526: CalledProcessError

During handling of the above exception, another exception occurred:

rejection_sampler = RejectionSampler(), k = 1, vocab_size = 1000, batch_size = 1, frac_seeded = 0.0, n_rep = 20

    @pytest.mark.parametrize("k", [1, 3, 5])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("frac_seeded", [0.0, 0.5])
    @pytest.mark.parametrize("n_rep", [20])
    def test_deterministic_when_seeded(
        rejection_sampler,
        k: int,
        vocab_size: int,
        batch_size: int,
        frac_seeded: float,
        n_rep: int,
    ):
        num_tokens = batch_size * k
        draft_probs = torch.rand(num_tokens,
                                 vocab_size,
                                 dtype=torch.float32,
                                 device=DEVICE)
        draft_probs = F.softmax(draft_probs, dim=-1)
        target_logits = torch.rand_like(draft_probs)
        bonus_token_ids = torch.randint(low=0,
                                        high=vocab_size,
                                        size=(batch_size, 1),
                                        dtype=torch.int64,
                                        device=DEVICE)
        draft_token_ids = torch.randint(low=0,
                                        high=vocab_size,
                                        size=(batch_size, k),
                                        dtype=torch.int64,
                                        device=DEVICE)

        seeded_mask = torch.rand(batch_size, dtype=torch.float32) <= frac_seeded

        results = []
        for _ in range(n_rep):
            seeded_seqs = {
                i: torch.Generator(device=DEVICE).manual_seed(i)
                for i in range(batch_size) if seeded_mask[i]
            }

            temperature = torch.ones(batch_size,
                                     dtype=torch.float32,
                                     device=DEVICE)
            sampling_metadata = create_sampling_metadata(all_greedy=False,
                                                         temperature=temperature,
                                                         generators=seeded_seqs)
            spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids.tolist(), device=DEVICE)
>           rep_result = rejection_sampler(
                spec_decode_metadata,
                draft_probs=draft_probs,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                sampling_metadata=sampling_metadata,
            )

tests/v1/sample/test_rejection_sampler.py:314:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
../../../.conda/envs/triton/lib/python3.10/site-packages/torch/nn/modules/module.py:1532: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
../../../.conda/envs/triton/lib/python3.10/site-packages/torch/nn/modules/module.py:1541: in _call_impl
    return forward_call(*args, **kwargs)
vllm/v1/sample/rejection_sampler.py:95: in forward
    output_token_ids = rejection_sample(
vllm/v1/sample/rejection_sampler.py:205: in rejection_sample
    recovered_token_ids = sample_recovered_tokens(
vllm/v1/sample/rejection_sampler.py:417: in sample_recovered_tokens
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
../../ascend/triton-ascend/triton/python/triton/runtime/jit.py:330: in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
../../ascend/triton-ascend/triton/python/triton/runtime/jit.py:623: in run
    kernel = self.compile(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

src = <triton.compiler.compiler.ASTSource object at 0xffffb3fe1000>, target = GPUTarget(backend='npu', arch='Ascend910B3', warp_size=0)
options = NPUOptions(debug=False, sanitize_overflow=True, llvm_version=15, kernel_name='triton_', cluster_dims=(1, 1, 1), num_wa...nput_precision='ieee', enable_npu_compile=True, max_num_imprecise_acc_default=None, extern_libs=None, multibuffer=True)

    def compile(src, target=None, options=None):
        if target is None:
            target = driver.active.get_current_target()
        assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
        backend = make_backend(target)
        ir_source = not isinstance(src, ASTSource)
        # create backend
        if ir_source:
            assert isinstance(src, str), "source must be either AST or a filepath"
            src = IRSource(src)
        extra_options = src.parse_options()
        options = backend.parse_options(dict(options or dict(), **extra_options))
        # create cache manager
        env_vars = get_cache_invalidating_env_vars()
        key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
        hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        fn_cache_manager = get_cache_manager(hash)
        # For dumping/overriding only hash the source as we want it to be independent of triton
        # core changes to make it easier to track kernels by hash.
        enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
        enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
        fn_override_manager = get_override_manager(src.hash()) if enable_override else None
        fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
        # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
        # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
        # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
        # the file name to 150 characters to be safe.
        file_name = src.name[:150]
        metadata_filename = f"{file_name}.json"
        metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
        metadata_path = metadata_group.get(metadata_filename)
        always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0") == "1"
        if not always_compile and metadata_path is not None:
            # cache hit!
            metadata = json.loads(Path(metadata_path).read_text())
            return CompiledKernel(src, metadata_group, hash)
        compile_speed_opt = os.getenv("TRITON_ASCEND_COMPILE_SPEED_OPT", 'false').lower() in ('true', '1')
        if (compile_speed_opt):
            ttir_path = f"{file_name}.ttir"
            if (metadata_path is None) and (fn_cache_manager.has_file(ttir_path)):
                # Already compile once but failed. So directly return
                raise Exception("already failed once")
        # initialize metadata
        metadata = {
            "hash": hash,
            "target": target,
            **options.__dict__,
            **env_vars,
        }
        # run compilation pipeline  and populate metadata
        stages = dict()
        backend.add_stages(stages, options)
        first_stage = list(stages.keys()).index(src.ext)
        # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
        if ir_source:
            first_stage += 1
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation()
        module_map = backend.get_module_map()
        try:
            module = src.make_ir(options, codegen_fns, module_map, context)
        except Exception as e:
            filter_traceback(e)
            raise
        use_ir_loc = os.environ.get("USE_IR_LOC", None)
        for ext, compile_ir in list(stages.items())[first_stage:]:
            try:
                next_module = compile_ir(module, metadata)
            except Exception as e:
                if (ext == "ttadapter"):
                    stage_name = "ConvertTritonIRToLinalgIR"
                elif (ext == "npubin"):
                    stage_name = "ConvertLinalgRToBinary"
                else:
                    stage_name = "MLIRCompile"
>               raise MLIRCompilationError(stage_name, e.stderr.decode('utf-8'))
E               triton.compiler.errors.MLIRCompilationError:
E               ///------------------[ERROR][Triton][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/v1/sample/rejection_sampler.py:569:0: error: 'func.return' op has 1 operands, but enclosing function (@sample_recovered_tokens_kernel) returns 0
E               /home/devuser/workspace/vllm-project/vllm/vllm/v1/sample/rejection_sampler.py:569:0: note: see current operation: "func.return"(%12) : (i1) -> ()
E               ///------------------[ERROR][Triton][END]------------------

../../ascend/triton-ascend/triton/python/triton/compiler/compiler.py:297: MLIRCompilationError
============================================================================================================== warnings summary ==============================================================================================================
../../../.conda/envs/triton/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:59
  /home/devuser/.conda/envs/triton/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:59: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/latest owner does not match the current owner.
    warnings.warn(f"Warning: The {path} owner does not match the current owner.")

../../../.conda/envs/triton/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:59
  /home/devuser/.conda/envs/triton/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:59: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
    warnings.warn(f"Warning: The {path} owner does not match the current owner.")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================================================================================================== short test summary info ===========================================================================================================
FAILED tests/v1/sample/test_rejection_sampler.py::test_deterministic_when_seeded[20-0.0-1-1000-1] - triton.compiler.errors.MLIRCompilationError:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
================================================================================================ 1 failed, 16 deselected, 2 warnings in 8.11s ================================================================================================
```
</details>

### vllm/lora/ops/triton_ops/kernel_utils.py

- [ ] `mm_k`
- [ ] `do_expand_kernel` —— 会调用 `tl.max_contiguous`，`tl.multiple_of`
- [ ] `do_shrink_kernel` —— 会调用 `tl.max_contiguous`，`tl.multiple_of`，一定条件下会调用 `tl.atomic_add`

> 触发条件：CUDA

### vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py

- [ ] `_chunk_scan_fwd_kernel`

> 触发条件：CUDA

### vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py

- [ ] `_chunk_cumsum_fwd_kernel` —— 会调用 `tl.cumsum`
- [ ] `_chunk_state_fwd_kernel`
- [ ] `_chunk_state_varlen_kernel`

> 触发条件：CUDA
