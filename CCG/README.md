# CCG Image Analysis
## WSL Setup
Search for "PowerShell" and open it in administrator mode. <BR>

Enter the command ```wsl --install``` and then restart your machine

Open VS Code and click on ![image2](https://github.com/Global-Health-Labs/LFA_GUI/blob/analysis_auto/readme_images/remote_explorer.PNG) icon to open a new window with Ubuntu.

## MicroMamba Setup
Start your VS Code and open up the Ubuntu terminal (Ctrl-Shift-~). <BR>

Get the executable using the code below: <BR>
```wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh``` <BR>
Execute the file by running the code: ```bash Mambaforge-Linux-x86_64.sh```<BR>

Restart your machine to complete the setup.

# Clone Repository
Copy the link below:<BR>
```https://github.com/Global-Health-Labs/LFA_GUI/```

Open VS Code and go to the "Source Control" tab (Ctrl-Shift-G) on the left navigation bar in VS Code and click the "Clone Repository" button. 
![image3](https://github.com/Global-Health-Labs/LFA_GUI/blob/analysis_auto/readme_images/clone_repo.png)<BR>
Paste in the URL you copied and press the "Enter" key on your keyboard

# Environment Setup
Navigete to the CCG Repository:<BR>
```cd CCG/CCG_image_analysis```

Run the code below to setup the environment:<BR>
```conda env create -f ccg_analysis_environment.yml```

Activate the environment using ```conda activate ccg_img_analysis```
<BR>

# Workflow
Once you have the images, crop them using ```cropper.py``` using the command below:<BR>
```python3 cropper.py /path/to/input_dir /path/to/output_dir```

Verify the cropping worked and rename the files into "CCGxx_xxx" format. For example: "CCG4_012"

Then run ```ccg_sorter.sh``` to sort all the files into their resective CCG module numbers:<BR>
```bash ccg_sorter.sh /path/to/source_dir /path/to/dest_dir```

After sorting them, we will stitch all the individual crops together as a single image using ```stitcher.sh```:
```bash stitcher.sh /path/to/source_dir /path/to/dest_dir```
