#include <iostream>

typedef int processId;

mutex sameTypeLock;
mutex oppositeTypeLock;

int numSecret = 0;
int numOpen = 0;

mutex_init(&sameTypeLock, NULL);
mutex_init(&oppositeTypeLock, NULL);


void Remove(processId id)
{
    printf("process %d has been removed", id)
}

processId Use(int itemType)
{  
    
    if(itemType == 0)
    {
        acquire(sameTypeLock);
        numSecret++;

        if(numSecret == 1)
            acquire(oppositeTypeLock);
        
        release(sameTypeLock);

        printf("do cool stuff");

        numSecret--;
        if(numSecret == 0)
            release(oppositeTypeLock);

        release(sameTypeLock);
        Remove(myID);
    }
    else if(itemType == 1)
    {
        acquire(sameTypeLock);
        numOpen++;

        if(numOpen == 1)
            acquire(oppositeTypeLock);
        
        release(sameTypeLock);

        printf("do cool stuff");

        numOpen--;
        if(numOpen == 0)
            release(oppositeTypeLock);
            
        release(sameTypeLock);
        Remove(myID);
    }

    return myID;
}