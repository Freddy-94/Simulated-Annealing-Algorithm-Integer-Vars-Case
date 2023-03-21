# Simulated-Annealing-Algorithm-Integer-Vars-Case (Quadratic assignment problem)

This repository will contain an implementation of the simulated annealing algorithm, for an integer variables case.
Our program adjust the simulated annealing algorithm presented here: https://github.com/Freddy-94/Simulated-Annealing-Algorithm-Real-Vars-Case to the case where we consider a cost function of Integer variables, and provide approximate solutions for the quadratic assignment problem (https://en.wikipedia.org/wiki/Quadratic_assignment_problem), for problems of size: 9!, 12!, 15!, and 30!

## Quadratic Assignment Problem

https://en.wikipedia.org/wiki/Quadratic_assignment_problem

![imagen](https://user-images.githubusercontent.com/36865111/226516495-e18497e4-b83f-49a9-8b79-ea4944ae3a72.png)


## Mutator operators

In order to adjust our previous algorithm, we implement 3 different "mutation" operations that generate a new neighbor of the random start solution. This operataions are:

1. Swap operator: Two random indexes of the solution vector are taken, and their corresponding values are swaped:

![imagen](https://user-images.githubusercontent.com/36865111/226515518-c34d523f-af41-4569-a683-df4a7c84dec3.png)

2. Reverse operator: Two random indexes of the solution vector are taken, and the values of this "sub-vector" are reversed:

![imagen](https://user-images.githubusercontent.com/36865111/226515480-6f06a368-6ef5-4995-a1aa-846736631ece.png)

3. Insersion operator: A random element of the vector, along with a random index are taken, and the random element taken is inserted in the random index taken, moving the other elements of the vector:

![imagen](https://user-images.githubusercontent.com/36865111/226515447-8269e46d-8d80-4d4a-af25-1d7c9e97500c.png)


## Execution of the program

The program needs a configuration file that needs to be passed in the terminal. The data in this file consists of the flux and distance matrices. Now, if for example, this config file is called "tai15.dat", then, you only need to locate in the directory where the QAP module and the mentioned config file is located, and run the command: 

           python QAP.py < tai9.dat

## Relevant functions:

1. swap_operator -> swap operator
2. reverse_operator  -> reverse operator
3. insersion_operator   -> insersion operator
4. cost_function_perm -> cost function implementation described above
5. acceptance_probability -> Acceptance probability criteria
6. see_annealing -> Plot solutions, costs, vs. number iterations, respectively 
7. annealing -> Simulated annealing algorithm

Example of results obtained with the SWAP operator, for a problem of 9! size:

![imagen](https://user-images.githubusercontent.com/36865111/226516227-a9d5d196-23e3-4f21-9cb9-697f76c182eb.png)


