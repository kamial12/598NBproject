Get data files from: https://github.com/DistriNet/DLWF

Environment info (i.e. packages needed and versions that work for me)
I made a virtual environment with:
python version 2.7.14
pip version 9.0.1
Packages to install (with versions that I have):
configobj (5.0.6)
statistics (1.0.3.5)
numpy (1.14.2)
keras (2.1.6)
tensorflow (1.7.0)
hyperas (0.4)
matplotlib (2.2.2)

cd keras-dlwf/

Edit tor_create_conf.py (if needed)
Things you might need to edit:
config[‘dnn’] should be ‘cnn’, ‘sdae’, or ’lstm’ depending on which type of model you want to train
cnn_config, sdae_config, lstm_config should be the path to the training set file (change for your data folder)
config[‘test_data’] should be the path to the test data (change for your data folder)

After editing, to generate a new tor.conf file (used in main.py), run:
python tor_create_conf.py

Then to train the model and run the eval, run:
python main.py


Dataset Info:
Closed
tor_900w_2500tr.npz
Same as CW900 in paper
Data dimensions (900*2500) by 5000
=(number of websites*number of valid network traces each) by (size of network trace?)
Labels
(900*2500)
labels[i] is the website label for data[i,]
len(set(labels)) = 900 i.e. 900 different websites so 900 unique label choices

tor_200w_2500tr.npz
Same as CW200 in paper
Dataset for the top 200 websites
Data dimensions (200*2500) by 5000
=(number of websites in ds * number of valid network traces each) by (size of network trace?)
Labels
(200*2500)
len(set(labels))=200

Open world
2 datasets
tor_open_400000w.npz
Dataset for top 400,000 Alexa websites
Single instance for each page

tor_open_200w_2000tr.npz
Additional 2000 test traces for each website of the monitored closed world CW200 (described above)
400,000 instances here (2000 test traces * 200 websites)

In the paper, the open world evaluation is on all 800,000 test traffic traces, half closed world (tor_open_200w_2000tr.npz) and half open world (tor_open_4000000w.npz)
