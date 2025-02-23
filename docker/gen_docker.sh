set -e
docker build --network host -t sglang_eic_dev -f Dockerfile.eic ../..
docker tag sglang_eic_dev eic2-cn-beijing.cr.volces.com/eic-image/eic_test:sglang044_dev3
docker push eic2-cn-beijing.cr.volces.com/eic-image/eic_test:sglang044_dev3