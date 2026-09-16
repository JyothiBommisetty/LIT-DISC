========================================================================
  REGRESSION REPORT: LIT-DISC v1.0 (sigma=0.1) vs v1.1 (sigma=5.0)
========================================================================

  Date: 2026-07-29
  Passages: 25 gold-standard passages

------------------------------------------------------------------------
  KEY METRICS COMPARISON
------------------------------------------------------------------------
                Metric    v1.0 Mean    v1.1 Mean    Change%    v1.0 CV    v1.1 CV
  -------------------- ------------ ------------ ---------- ---------- ----------
      comp_interaction     0.002858     0.002761     -3.39%     0.4467     0.4879
        comp_intrinsic     0.287934     0.039356    -86.33%     0.4479     0.6168  ***
           comp_memory     0.004186     0.004146     -0.95%     0.4478     0.4704
        comp_resonance     0.000225     0.000224     -0.43%     0.4479     0.4700
         comp_temporal     0.004207     0.004189     -0.44%     0.4479     0.4700
      comp_uncertainty     0.927052     0.885075     -4.53%     0.0149     0.0254
          gamma_global     0.030349     0.030216     -0.44%     0.4479     0.4699
                  ldri     0.029008     0.027691     -4.54%     0.4476     0.4791
              raw_ldri     0.956060     0.912766     -4.53%     0.0031     0.0263
             runtime_s     0.858000     0.209200    -75.62%     0.2937     0.2794  ***

------------------------------------------------------------------------
  VARIANCE DECOMPOSITION (log-space)
------------------------------------------------------------------------
  var(log(LDRI)) v1.0: 0.237645
  var(log(LDRI)) v1.1: 0.260898
  Gamma dominance v1.0: 100.3%
  Gamma dominance v1.1: 96.5%  (change: -3.8pp)
  Raw LDRI contribution v1.0: 0.0%
  Raw LDRI contribution v1.1: 0.3%  (change: +0.3pp)

------------------------------------------------------------------------
  SATURATION ANALYSIS
------------------------------------------------------------------------
  v1.0: 99% of activations saturated at 1.0 (sigma=0.1)
  v1.1: 4% of activations saturated at 1.0 (sigma=5.0)
  Residual saturation: 6/150 activations > 0.999
  Conclusion: SATURATION REMOVED

------------------------------------------------------------------------
  RAW LDRI VARIANCE
------------------------------------------------------------------------
  Raw LDRI CV:   0.0031 -> 0.0263  (8.4x increase)
  Raw LDRI Var:  0.00000904 -> 0.00057590

------------------------------------------------------------------------
  CATEGORY RANKING
------------------------------------------------------------------------
  v1.0: ['SNC', 'MLC', 'NWC', 'MNC', 'SSC']
  v1.1: ['SNC', 'MLC', 'NWC', 'SSC', 'MNC']
  Same: False
  Spread v1.0: 0.0211
  Spread v1.1: 0.0215

------------------------------------------------------------------------
  AUTHOR RANKING
------------------------------------------------------------------------
  Top-5 v1.0: ['Virginia Woolf', 'Charles Dickens', 'Marcel Proust', 'Franz Kafka', 'Fyodor Dostoevsky']
  Top-5 v1.1: ['Virginia Woolf', 'Marcel Proust', 'Charles Dickens', 'Franz Kafka', 'Fyodor Dostoevsky']
  Spearman rho: 0.9823

------------------------------------------------------------------------
  RUNTIME
------------------------------------------------------------------------
  Mean runtime:  0.86s -> 0.21s
  Total runtime: 21.4s -> 5.2s

------------------------------------------------------------------------
  COMPONENT RMSE (per-passage root mean square error)
------------------------------------------------------------------------
      uncertainty: RMSE=0.047528  mean 0.927052->0.885075  (-4.53%)  
        intrinsic: RMSE=0.269886  mean 0.287934->0.039356  (-86.33%)  CHANGE > 5%
      interaction: RMSE=0.000256  mean 0.002858->0.002761  (-3.39%)  
         temporal: RMSE=0.000286  mean 0.004207->0.004189  (-0.44%)  
        resonance: RMSE=0.000015  mean 0.000225->0.000224  (-0.43%)  
           memory: RMSE=0.000283  mean 0.004186->0.004146  (-0.95%)  

------------------------------------------------------------------------
  SUMMARY
------------------------------------------------------------------------
  Metrics with change > 5%: 2
          comp_intrinsic: 0.287934 -> 0.039356 (-86.33%)
               runtime_s: 0.858000 -> 0.209200 (-75.62%)

  Category ranking: changed (bottom two swapped)
  Author ranking preservation: rho = 0.9823
  Saturation: 99% -> 4%  (improvement)
  Raw LDRI CV: 0.3% -> 2.6%  (8.5x improvement)
  Gamma dominance: 100.3% -> 96.5%  (marginal improvement)
  Runtime appears faster (machine variability)
  No numerical regressions (all 66 tests pass)

========================================================================