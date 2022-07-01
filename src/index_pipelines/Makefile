jupyter-notebook:
	srun \
		--container-name=ecir22-pairwise-jupyter \
		--mem=25G -c 2 \
		--container-image=jupyter/datascience-notebook \
		--container-writable \
		--container-mounts=/mnt/ceph/storage/data-tmp/2021/kibi9872/sigir22-pairwise-ranking:/workspace,/mnt/ceph/storage/data-in-progress/data-research/web-search/SIGIR-22/sigir22-pairwise-ranking/ \
		--pty \
		bash -c 'cd /workspace && jupyter notebook --ip 0.0.0.0 --allow-root'

