# Disaster-Response-Pipelines-Project
Udacity Project for Data Science Nanodegree 

---------------------------------------------------------

**Project motivation:** the purpose of this project is to analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. This will help disaster response organizations in two ways:
1. Filter and pull out important messages
2. Sort disaster messages properly to provide prper aid 

This is done by text processing for the messages and utilizing supervised machine learning model to classify the message, for example, if water disaster occured and someone wrote am thirsty, this should be classified as water disastar and repsonded to it as such

----------------------------------------------------------

**Requirements:**
The following libraries are included:
* Pandas
* numpy
* nltk
* skilearn
* flask
* sqlalchemy
* re
* pickel

----------------------------------------------------------

**Files in the repository:**
* [model]: machine learning pipeline, which I built
* [data]: contains 2 csv files + ETL pieline, which I built
* [app]: contains 2 html files + flask file, which I modefied to have two addtional visualization. One that shows the top 5 disaster categories and the other that shows the lowest 5 disaster categories.

*data used:* given as csv file from Figure Eight

-------------------------------------------------------------------

**Acknowledgement:**
This project was built over the data availed from Figure Eight. You can know more about them through the following [link](https://en.wikipedia.org/wiki/Figure_Eight_Inc.) 
