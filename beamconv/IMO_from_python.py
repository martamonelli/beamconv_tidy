import json
import numpy as np

print('----------------------------------------------------\n')
print('preliminary stuff...\n')
  
# opening IMO schema.json file and interpreting it as a dictionary
f = open('litebird_imo-master/IMO/schema.json',)
data = json.load(f)  

# looking into the IMO, data['data_files'] is where the relevant info is stored 
data_files = data['data_files']

# counting how many objects are in the dictionary
nkey=0
for key in data_files:
    nkey = nkey+1

print('detectors belonging to the L2-050 channel:')

# looking for the detectors belonging to the L2-050 channel
# SHOULD I USE ANOTHER CHANNEL?
for i in np.arange(nkey):
    test = data_files[i]
    if(test['name'] == 'channel_info'):
        metadata = test['metadata']
        if(metadata['channel'] == 'L2-050'):
            detector_names = metadata['detector_names']
            break

print(str(detector_names)+'\n')

ndet = len(detector_names)

list_of_dictionaries = []

# looking for the metadata of the detectors in detector_names
for d in detector_names:
    #print(d)
    for j in np.arange(nkey):
        test = data_files[j]
        if(test['name'] == 'detector_info'):
            metadata = test['metadata']
            if (metadata['name'] == d):
                list_of_dictionaries.append(metadata)
                #print(metadata['name'])
                break
            #break
        #break

print('now I have a list of dictionaries with the metadata of each detector!\n')

print('for instance, the 3rd element contains the metadata of detector '+detector_names[2]+':')
print(list_of_dictionaries[2])

#for j in np.arange(ndet):
#    print(j)
#    print('metadata of detector '+detector_names[j]+':')
#    print(list_of_dictionaries[j])

print('')

print('is this okay?')

print('\n----------------------------------------------------')
