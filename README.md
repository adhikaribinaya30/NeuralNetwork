# IMPORTANT NOTE: I HAVE VERY LITTLE IDEA ABOUT THESE TOPICS (on 2026/05/11) SO WHAT I HAVE 
#                 WRITTEN IN THE EARLIER DAYS MIGHT BE WRONG. AS I GO ON LEARNING MORE ABOUT EACH TOPIC I WILL 
#                 TRY TO CORRECT THEM TO THE BEST OF MY ABILITIES 

# NOTE: File names with ....-self.py are the files that I wrote on my own at the end to revise

Day 1:
  I researched some stuff about ML and got to know about the four types of ML models
  
  1. Supervised learning
     -> It is the model which is trained using data and the outputs that are labelled meaning every input is labelled with its correct output
  
  2. Unsupervised learning
     -> In this learning model, the data provided is not labelled at all. The model gets no human guidance at all
  
  3. Semi-supervised learning
     -> In this the learning model is given a limited about of data that is labelled (input and corresponding output are mapped) and the majority of given data is unlabelled
     -> The model trains itself on the data and uses it to label the remaining data and trains itself on it
     -> Since labelling data manually takes up a lot of time this approach saves human effort
  
  4. Reinforcement learning
     -> In this ML model, the model doesn't receive any data
     -> It directly works in an environment to give results and gets results (Reward or Punishment) according to how it performed
     -> Then the model corrects itself every iteration to get more reward until it gets to the required output

Day 2:

  I found a video of the renowned Andrej Karpathy on ML. I watched his video where he talks about how a micrograd works step-by-step
  While I knew python basics I still hadn't gotten the habit of using Object Oriented Programming. So I decided it would be better to 
  get a hang of using Objects in python before starting to code the micrograd. 

  So I warmed up by building a very simple Library Management system using OOP


Day 3:

  I officially started learning how the micrograd worked. I learned about the Value class and how it managed nodes and stored
  the value of each node, their children and the operation required to come to that value. This allowed for using those
  information to travel back to all children nodes of the result nodes.

  Finally, using topological sort we sort the data in such an order that before a node is visited all of its 
  dependencies are first visited. Then we used this sorted data to apply backpropagation through any operation and
  find how much each value used in building up to the value affected it (using derivative of the result w.r.t the specific value)

  Then I also learned about pytorch and how the library can be used to automate the entire process.

  Earlier, while using topological sort, I had copy-pasted the logic for the code. So finally to wrap up the day
  I figured out how the sorting mechanism works and wrote it from scratch on my own.

  Commit 1: message 'Value object of micrograd, topological sort and backpropagation'
