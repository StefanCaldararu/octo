# Octo training setup on curc

Create a CURC account and ensure you can ssh into your account: 
[login docs](https://curc.readthedocs.io/en/latest/access/logging-in.html)

Once you are able to ssh in to a login node, clone octo:

`git clone https://github.com/octo-models/octo.git`

Begin by creating a `.condarc` file, and copying the following text into it:

```bash
pkgs_dirs:
  - /projects/$USER/.conda_pkgs
envs_dirs:
  - /projects/$USER/software/anaconda/envs
```

Login to a compile node, load the anaconda module, and create & setup a conda environment. The anaconda setup documentation for CURC can be found [here](https://curc.readthedocs.io/en/latest/software/python.html), or just follow these instructions:


```bash
acompile #Login to a compile node
cd octo #go into the octo directory
ml anaconda #load the anaconda module
conda create -n octo python=3.10 #create your environment
conda activate octo #activate the environment
pip install -e . #install octo
pip install -r requirements.txt #install the octo requirements
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # install jax for gpu
exit #disconnect from the compile node
```

You should now be ready to run the slurm job and test the octo debug script! This can be done with `sbatch octo_testjob.sh` once you have grabbed the script from this repo. I have been able to run this on CURC with no errors. There are a few notes in relation to this though:

- Make sure to change line 24 to source `cublas` from your own python environment, as you likely don't have access to mine.
- If you want to run a job with your own fairshares rather than the `ucb510_asc1`, remove line 3 from the script.
- Modify the time to be 1 hour if you are just running the debug script, as this will schedule your job sooner.
- If you want to run the finetuning example, uncomment that line and comment the debug line. I am currently experiencing an error that I didn't see previously that the aloha sim data can't be loaded. 
