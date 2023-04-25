import random
task_to_class_list = {"1":[], "2":[], "3":[], "4":[], "5":[]}
tasks = [1, 2, 3, 4, 5]
for i in range(0, 200):
    splits = []
    splits.append(random.choice(tasks))
    for j in tasks:
        r = random.random()
        if r < 0.1:
            splits.append(j)
    for j in splits:
        task_to_class_list[str(j)].append(i)

print(task_to_class_list)        