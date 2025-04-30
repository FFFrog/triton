# Triton Ops in vLLM

## Env

```
torch                             2.3.1
torch-npu                         2.3.1.post2
inductor_npu                      0.1
triton-ascend                     36969dafdad51877233f6adb2c077212d5058f1d
```

### vllm/attention/ops/chunked_prefill_paged_decode.py

- [x] `cdiv_fn`
- [x] `kernel_paged_attention_2d`

```python
pytest -svx tests/kernels/attention/test_prefix_prefill.py -k "chunked_prefill_paged_decode"
```

<details>
<summary>HuaweiCompilationError</summary>

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
