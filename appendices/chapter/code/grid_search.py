import itertools as it

param_grid = {
    'max_depth' : [2,4,8,16,32,64],
    'learning_rate' : [0.1,0.01,0.001, 0.0001],
    'batch_size' : [16,32,64, 128]
    }
combinations = it.product(*(param_grid[param] for param in param_grid))

List1_ = []
for i in combinations:
    List1_.append(i)
print ('# of combinations: {}'.format(len(List1_)))

with open('grid_search_{}.txt'.format(len(List1_)),'w') as f:
    for listitem in List1_:
       listitem = str(listitem)
       listitem = listitem.strip('()')
       listitem = listitem.replace(',','')
       f.write('%s\n'% listitem)

