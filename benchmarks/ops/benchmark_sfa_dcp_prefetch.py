# SPDX-License-Identifier: Apache-2.0
"""A/B next-layer H2D before packing versus alongside asynchronous all-to-all.

Both arms include real projections/current KV, resident planning, registered
host retrieval, DCP attention/merge and local output projection. Query-gather
timing and the 12-block prefetch width are identical in both arms.
"""

from benchmark_sfa_dcp_query_first import QueryFirstPipeline, main


def prefetch_pipeline(args, inputs, sources, group, candidate):
    pipeline = QueryFirstPipeline(args, inputs, sources, group, candidate=args.query_first)
    pipeline.prefetch_after_pack = candidate
    return pipeline


if __name__ == "__main__":
    main(comparison="prefetch-after-pack", pipeline_factory=prefetch_pipeline, description=__doc__)
