from random import randint
import math
import random as rd
import numpy

################# INITIALISE POPULATION #################

#Initialise population, of size popSize, of 1s & 0s of given chromosome size
def initSolution(chromoSize, popSize):              #Takes two Integers as arguments, a size of the chromosome(in this case its 32) and a population size for the GA
    solutions = []                                  #Initialise the array of individuals this function will return
    for i in range(popSize):                        #loop through to generate individuals
        solutions.append([randint(0, 1) for _ in range(chromoSize)])        #this line is generating random 1s or 0s for the size of the chromosome (32), these are our individuals
    return solutions
        
################# FITNESS FUNCTION ##################

    #This is our fitness function, this puts a rank or value to each individual to show how many '1's they have - more = better and higher fitness score = more chance of becoming parents

def fitnessFunction(individual):        #fitness is simply determined by counting the number of '1's in an individual
    count = 0
    for i in (individual):              #looping through individual and returning a 'count' representing the fitness or 'strength' of each individual in the population 
        if i == 1:
            count += 1
    return count                         #return the fitness score for an individual

    #Mutate Function takes an individual and a mutation rate (% chance that mutation will take place), and simply flips a bit if it is not a '1', there is a 10% chance of this happening in this case
    #This function is very important in determining how many generations / pop size it takes to come to a solution, my 1bit-flip is a very weak mutation so it takes a substantial amount of gens
    
def mutate(chromo, mutationRate):                               #takes a chromosome and a <1 mutation rate as arguments (usually 0.1)
    bitSelect = rd.randint(0, (len(chromo)-1))                  #random index chosen to be bit-flipped for successful mutation
    if (rd.random() <= mutationRate and chromo[bitSelect] == 0):#this is where the algoritm decides whether mutation happens or not based on the probability fed into this function, also to completely neglect the chance of reproducing a 0 which results in the algorithm never resolving to 32 '1's
        chromo[bitSelect] = 1 - chromo[bitSelect]               #bit being flipped, important as this is the mutation process taking place
    return chromo
        
def mating_crossover(parent_a,parent_b):                            #takes two parents and selects a point in them to flip round the :x from parentA with the :y from parentB
    offspring=[]
    cut_point= int(len(parent_a) / 2)
    offspring.append(parent_a[:cut_point] + parent_b[cut_point:])
    offspring.append(parent_b[:cut_point] + parent_a[cut_point:])
    return offspring
    
def rouletteWheel(fitness,pop):                 #both arguments are arrays in this case
    parents=[]
    fitTotal=sum(fitness)                       #adding all fitness scores together for normalisation
    normalised=[x/fitTotal for x in fitness]    #Roulette Wheel requires normalised values between 1-0 because these decimal values act the same as percentages later in the algorithm
    
    fCumulative=[]                              #declare cumulative array for selection process
    index = 0
    for nVal in normalised:
        index += nVal
        fCumulative.append(index)
        
#    print("Cumulative Fitness: ")
#    arr_fprint(fCumulative)
    
    popSize = len(pop)                      #declaring a manageable variable for when using the size of the population
    
    for _ in range(popSize):                  #looping through population
        rand_n=rd.uniform(0,1)                      #generate a real number between 1 and 0 (the pointer or spin of the wheel)
        individual_n=0                              #index for the population
        for fitvalue in fCumulative:
            if(rand_n<=fitvalue):                   #the random number(pointer of the wheel) stays the same in this nested for loop until a number higher than the random number is reached, meaning the chance of this loop making it all the way through fCumalative is very low.
                parents.append(pop[individual_n])   #appending the corresponding individual to parents, higher the fitness function of an individual = the higher the normalised value = higher chance% of selection
                break
            individual_n+=1
    return parents

def arr_fprint(array):
    for i in array: print(i)

chSize = 32             #32 bit chromosome
pSize = 30              #population size of 30, could be any number
gens = 800              #this program requires roughly >750 generations to see the genetic process of how it comes to a solution

#Here i have left a print statement showing the fitness values of each generation, easy to see the numbers increase to the desired 32

def ComputerRunProgram():                   #Computer, Run Program...
    pop = initSolution(chSize, pSize)
    
    for g in range(gens):                   #looping the amount of generations we declared just above
        fitValues = []          #initialise fitness array
        parents = []            #initialise parents array
        offSpring = []          #initialise offspring array
        mutatedPop = []         #initialise mutated population array for the n+1 generation to use as a population
        
        #print(g+1, "'s popualtion:")
        #arr_fprint(pop)
        
        for i in pop:
            fitValues.append(fitnessFunction(i))        #fitness values are calculated
            
        print("\nfitness values: ")
        for f in fitValues: print(f, end=", ")          #i have left fitness values printed because in the terminal you can clearly see all these fitness values reach 32, if they're not all 32 please increase the 'gens' variable
        

        parents = rouletteWheel(fitValues, pop)         #parents are selected by chance based off their fitness
        
        #print("parents:")              #print parents
        #arr_fprint(parents)

        for j in range(0, len(parents), 2):
            if j+1 < len(parents):
                offSpring.extend(mating_crossover(parents[j], parents[j+1]))    #offspring are made by crossing over parents by a cut-point in the middle of the parents
                
        #print("Offspring: ")           #print offspring
        #arr_fprint(offSpring)
        
        for c in offSpring:
            newC = mutate(c, 0.1)                   #mutating offspring using a 0.1(10%) mutation rate, mutation is a simple random bit swap if the bit is a 0
            mutatedPop.append(newC)
        
        pop = mutatedPop            #new mutated population is used for the next iteration(generation)
        
        #print("mutated population:")
        #arr_fprint(mutatedPop)
    print("\n Here we can see the fitness values for each individual in the population reach the value of 32 which indicates the individuals all have a count of 32 '1's in them")

ComputerRunProgram()
            
        



