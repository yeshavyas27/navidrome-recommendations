#!/usr/bin/env bash
# Manual rollback of the navidrome-serve deployment to a specific
# MinIO model key, overriding the auto-discover-latest behavior.
#
# Usage:
#   ./rollback.sh <minio_key>         # roll back to a specific model
#   ./rollback.sh --reset             # clear override, resume auto-latest
#   ./rollback.sh --current           # show what's running now
#
# Examples:
#   ./rollback.sh finetune/20260419/abc123/model.pt
#   ./rollback.sh pretrain/20260415/baseline/model.pt
#   ./rollback.sh --reset
#
# See TRIGGERS.md for when to invoke this.

set -euo pipefail

NAMESPACE="${NAMESPACE:-navidrome-platform}"
DEPLOY="${DEPLOY:-navidrome-serve}"

usage() {
    grep '^#' "$0" | head -20 | sed 's/^# \?//'
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

case "$1" in
    --reset)
        echo "Resetting MINIO_MODEL_KEY override on ${DEPLOY} — resumes auto-latest discovery."
        kubectl set env -n "${NAMESPACE}" "deploy/${DEPLOY}" MINIO_MODEL_KEY-
        ;;
    --current)
        echo "Running image:"
        kubectl describe pod -n "${NAMESPACE}" -l "app=${DEPLOY}" | grep "Image ID:" | head -1
        echo
        echo "Last model loaded (from pod logs):"
        kubectl logs -n "${NAMESPACE}" "deploy/${DEPLOY}" | grep -E "Loading model from|Downloading model" | tail -3
        exit 0
        ;;
    --help|-h)
        usage
        ;;
    *)
        MODEL_KEY="$1"
        # Basic sanity: must look like a MinIO key path with model.pt
        if [[ ! "${MODEL_KEY}" =~ .*model\.pt$ ]]; then
            echo "ERROR: model key must end in 'model.pt'. Got: ${MODEL_KEY}" >&2
            exit 1
        fi
        echo "Rolling back ${DEPLOY} to MINIO_MODEL_KEY=${MODEL_KEY}"
        kubectl set env -n "${NAMESPACE}" "deploy/${DEPLOY}" "MINIO_MODEL_KEY=${MODEL_KEY}"
        ;;
esac

echo "Restarting deployment..."
kubectl rollout restart -n "${NAMESPACE}" "deploy/${DEPLOY}"

echo "Waiting for rollout to complete..."
kubectl rollout status -n "${NAMESPACE}" "deploy/${DEPLOY}" --timeout=120s

echo
echo "Rollback complete. Current pod status:"
kubectl get pod -n "${NAMESPACE}" -l "app=${DEPLOY}"
echo
echo "Tail of new pod's startup log:"
kubectl logs -n "${NAMESPACE}" "deploy/${DEPLOY}" --tail=20
