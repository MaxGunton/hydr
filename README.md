# hydrophone

## Requirements
1. Python
   > Download latest stable at version at 
   > [python.org](https://www.python.org/downloads/)
2. Port Audio - ***(only needed if installation of this package fails because of 
                    pyaudio)***
   ```bash
      # On Ubuntu OS run the following:
      $ sudo apt install portaudio19-dev python3-pyaudio
   ```
## Installation
After downloading and installing python open the terminal and run the following:
```bash
# 1. Upgrade pip to latest version
$ python -m pip install --upgrade pip
```
```bash
# 2. Install anaconda (virtual environment manager)
$ pip install anaconda
```
```bash
# 3. Upgrade anaconda to latest version
$ conda update -n base -c defaults conda
```
At some stage you will likely need to initialize the shell you are using to recognize 
anaconda commands (anaconda will provide the instructions to do so)
```bash
# 4. Create a new virtual environment
$ conda create -n hydrophone python=3.7.13
```
```bash
# 5. Activate the newly created virtual environment
$ conda activate hydrophone
```
Following step may require installation of portaudio if pyaudio fails to install.  
```bash
# 6. Install this package
$ pip install git+https://github.com/MaxGunton/hydr.git
```

## Description
This package contains scripts used to analyze hydrophone deployments, this includes (but may not be limited to) the 
following functionalities:

1. create new project directory structure

Then manually add the serial numbers of the hydrophones to the `00_hydrophone_data` directory
   
2. generate a summary of the data captured by each hydrophone

2b. Then you may need to rename the files if they are using the hardware serial number instead of the SoundTrap serial number
    This sometimes happens and should be corrected *could write a script for this*, it would also need to update the name of the config file and summary.  
    (Actually it wouldnt if we place these into the directories).  Don't implement this yet**

3. Move the files that occur outside the deployment bounds into a subdirectory called `ignore`
   Should write a script to record the bounds and store them.  Use the existing directories in hydrophone_data as 
TODO: Should store all the data in a single file possibly a config file for easy parsing and then have a export method that pulls the data we want from the report
      Therefore various modules will read it in add their bit and then write it back and update the summary accordingly.  

4. set hydrophone coordinates
   

5. set event times (i.e. bounds and sync events) 
   

6. generate a map (kml file) that can be uploaded on google maps showing:
    - hydrophones positions - *DONE*
    - distance between hydrophones - *DONE*    
    - area of interest - *DONE*
      - compute area - *TODO*
    - hydrophone bounds (i.e. deployment/retreival) - *TODO*
    - triangulated blasts - *TODO*
    - heatmap of activity- *TODO*
    - show the events over time in animation - *TODO*

7. launch ML to scan the audio

7. clean detections by (any subset of the following):
   - combining all individual detection files into a single file - *DONE*
   - drop entries that occur outside of bounds - *DONE*
   - removing background detections with >50% confidence - *DONE*
   - combining detections with same code that occur adjacent in time - *DONE*
   - using the confidence values compute score for each detection - *DONE*
   - sort detections with most likely blasts first - *DONE* (May want to delay this and use additional information 
     derived from the following steps)
   
7. measures detection characteristics - *TODO*
   - assign likely blast peak using peak pick - *TODO*
   - signal-to-noise ratio - *TODO*
   - peak frequency - *TODO*
   - peak value at frequency commonly associated to blasts - *TODO*
   - ...
   
8. using a combination of score and measured characteristics sort the detections in order of most likely True positives
   first
   
9. launch validation window

10. combine overlapping

11. resolve any multicodes

12. create blast statistic plots

13. run multilateration script
   
  
---

### 1. Create New Project Directory
Create a project directory structure for a new hydrophone deployment.  The `name` parameter becomes the name of the 
directory and the `dest` parameter is the directory in where it will be written (defaults to '.').     
```bash
$ new-deployment [-h] [--d DEST] name
```

---

### 2. Generate Hydrophone Summary
> **Warning**
> 
> This script expects that hydrophones used are "Ocean Instrument SoundTraps" and that the directory structure and 
> naming convention for them is `.../<SN>/<SN>.yymmddHHMMSS.<extension>`.  It also assumes that audio files are in 
> `wav` format and that their corresponding log files share the same basename except with the extension `log.xml`. Any 
> deviation from these assumptions may cause inaccurate summaries or script failure.

Generate a summary for a hydrophone.  This includes the files generated, configuration details such as gain and sample 
rate, and calibration data pull from the SoundTrap website.  The `datadir` parameter is the directory containing data 
for a single hydrophone and `dest` parameter is the directory where the summary will be written to (defaults to '.').  
```bash
$ hydrophone-summary [-h] [--d DEST] datadir
```

---

### 3. Set Hydrophone Coordinates