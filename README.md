# Scalable_Time_Series
Final Project creating a unique and tuned model for each of 145,000 different webpage forecasts

The data was far too large to be able to upload, but the code is available.

The final presentation summarizing the project and my findings can by downloaded [here](https://github.com/leaferickson/Scalable_Time_Series/blob/master/Final_Presentation.pptx)

Initial manual modeling is done in models folder, with the _sample .py scripts as my initial thrusts. Each model then recieved a rules set of its own, which the other scripts are for. These scripts contain the unique feaeture selection process for each model.

This process was then compiled into the script ["one hierarchy"](https://github.com/leaferickson/Scalable_Time_Series/blob/master/one_hierarchy.py), which tests each model for the given group of pages, selects that which performs the best, and then makes a forecast using that model.

The project was then completed using a data pipeline that I set up on AWS, including use of Amazon Lambda for computation.
