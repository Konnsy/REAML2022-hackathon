This code is an example solution of task 1 and is implementing the code in "task1_code_frame".

changes to the code frame:

• the classifier is implemented as 
	1. 4 blocks of [conv -> relu -> conv -> max_pool] to determine local features
	2. a global maximum pooling to get from arbitrary local shapes to a global and fixed-size shape
	   with the side effect of better generalization
	   (as mentioned in https://github.com/Konnsy/REAML2022-hackathon/wiki/Useful-code-snippets)	  
	3. fully connected -> relu -> fully connected to process global information
	
• the learning rate was set to 5E-5
• all other things, including the preprocessing are the same as in the frame code