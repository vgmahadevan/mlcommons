# Set up python 3.10 on rivanna

* We assume you have python 3.10.4 installed 
* We assume if you like to use the automated report generated (under development) you have  
  full latex installed

  
```bash
python3.10 -m venv ~/ENV3
source ~/ENV3/bin/activate
mkdir ~/cm
cd ~/cm
pip install cloudmesh-installer
cloudmesh-installer get sbatch
cms help
```

2. Generating experiment configurations

# chose a PROJECT_DIR where you like to install the code

Pick a project directory and generate make file.

```bash
export PROJECT_DIR=/project/mlcommons
mkdir -p ${PROJECT_DIR}
cd ${PROJECT_DIR}
export EQ_VERSION=latest
git clone ssh://git@github.com/laszewsk/mlcommons.git
cd mlcommons/benchmarks/earthquake/${EQ_VERSION}/experiments/ubuntu

# build slurm scripts
# cms sbatch generate --source=rivanna.in.slurm --config=rivanna-2epoch.yaml --name="project" --noos 
make project


# Generate the submit scripts
# cms sbatch generate submit --name="project.json" > job-project.sh
make generate
```

It's strongly advised that you inspect the output of the above to
validate that all generated scripts and files are correct.  Most jobs
take several hours, so correcting errors by inspecting the output will
save time when troubleshooting.

```bash
emacs project/card_name_a100_gpu_count_1_cpu_num_6_mem_64GB_TFTTransformerepochs_2/slurm.sh
```

or simply call

```bash
make inspect
```

Note: to exit emacs, press `Ctrl+x` and then `Ctrl+c` to return to your normal prompt.

3. Running the experiments

If all the output from above looks correct, you can execute the jobs
by running the last two scripts that are generated by cms sbatch
generate submit.



```bash
# sh job-project.sh
make run
# Submitted batch job 12345678
```

The number will be different for you.

To find out the status you can
do the following commands. The first looks up the job by the id, the second will list all jobs you submitted. if you just have one job it will return just that one job. `make status` is a shortcut to see all jobs of a user

```bash
squeue --job 12345678
squeue | fgrep $USER
make status
```


The `make run` will submit the job to slurm, and the
notebook file will be outputted in the
`$(pwd)/project/<experiment_id>` directory.

You can see the progress of each job by inspecting the `*.out` and
`*.err` files located in the `$(pwd)/project/<experiment_id>`).

A copy of the final notebook is placed in the slurm expeeriments
folder with the suffix `*_output.ipynb`, that can be inspected for
further details.

To watch the output dynamically

```bash
tail -f  project/card_name_a100_gpu_count_1_cpu_num_6_mem_64GB_TFTTransformerepochs_2/*12345678.out
```



### Generate Report

```bash
pdflatex report.tex
pdflatex report.tex
# bibtex report.bib
pdflatex report.tex
```

This will create a pdf named `report.pdf`.  You can download this to
your local to view the output to view the report generated as a result
of the exectuion.
