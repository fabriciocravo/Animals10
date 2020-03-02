Animals10:
ResNet_1000_Acc_Test => ResNet was fooled with an 1000 pixel perturbation 
ResNet_1000_Clean => Images from  ResNet_1000_Acc_Test where the attack was sucessfull 
ResNet_1000_3ModelClean => Images from  ResNet_1000_Acc_Test where the models ResNet, VGG, and AlexNet were sucessfully tricked
ResNet_2000_Filter_Acc_Test => Images generated with 2000 pixel perturbation created by minimizing the prediction confidence with a filter
All the other generated databases have not been discussed on the paper, however one can guess how they were generated by their name

Scripts:
	Main Folder
AlexNet.py => Trains AlexNet and allows the use of AlexNet model
differential_evolution.py => Differential evolution implementation by the creators of One Pixel Attack, not used here
GoogLeNet.py => Trains GoogLeNet and allows the use of GoogleNet model
HandNet.py => Hand crafted network, not used
PyTorchAttack.py => Generation of adversarial images by using the attacks implementation on PyTorch
ResNet.py => Trains ResNet and allows the use of ResNet model
SmallPerturbationAttack.py => Attempt to mimic the small perturbation attack from differential evolution. DO NOT RUN THIS, IT WILL CRASH YOUR COMPUTER.
VGG.py => Trains VGG and allows the use of VGG model

	DataAnalysis
Firstly to use the scripts in this folder it is necessary to download the produced the adversarial dataset
Afterwards set the path parameters accordingly to their localisation.
3ModelClean.py => Creates the ResNet_1000_3ModelClean dataset by removing the images predicted correctly by any of the three listed models
ClassAccuracyTest.py => Measure the class distribution for the attacked images
FilterTest.py => Measure the performance of different filters on adversarial data
MatchingClassifications.py => Measure the percentage of matching classification between the other models and ResNet
PercentageOfSucessfullAttacks.py => Measure fooling percentages and the accuracy of models on adversarial settings
Remove_Unsucessfull_Attacks.py => Creates the ResNet_1000_Clean by removing the unsucessfull attacks to ResNet






