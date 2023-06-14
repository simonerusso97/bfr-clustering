#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc50-cpp"
#pragma ide diagnostic ignored "cert-msc51-cpp"
#pragma ide diagnostic ignored "cert-err34-c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "mpi.h"

#define MEM_ALLOC_ERROR (-1)
#define FILE_OPENING_ERROR (-2)
#define INIT_ERROR (-3)
#define FILE_READING_ERROR (-4)
#define NOT_ENOUGHT_MEMORY (-5)
#define NOT_ABLE_TO_CLUSTER (-6)
#define WRONG_ARGUMENT (1)

#define READ_ONLY "r"
#define APPEND_MODE "a"

#define DIMENSION 2

#define MASTER_PROCESS 0
#define OUTPUT_FILE "output.csv"
#define OUTLIER_FILE "temp.csv"

typedef struct {
    int id;
    double coords[DIMENSION];
} Point;

typedef struct {
    int id;
    int size;
    double variance[DIMENSION];
    double coord[DIMENSION];
    double squared_coord[DIMENSION];
} Cluster;

typedef struct {
    int id;
    int size;
    double variance[DIMENSION];
    double coord[DIMENSION];
    double squared_coord[DIMENSION];
    Point *points;
} CompressedSet;

typedef struct {
    int idPoint;
    int idCluster;
} ClusteredPoint;

typedef struct {
    ClusteredPoint *clusteredPoint;
    Point *outlier;
    int numClusteredPoint;
    int numOutlier;
    int err;
} ClusteringResult;

typedef struct {
    Cluster *cluster;
    Point *outlier;
    ClusteredPoint *clusteredPoint;
    int numCluster;
    int numOutlier;
    int numClusteredPoint;
    int err;
} GNCResult;

typedef struct {
    Point *points;
    int numPoints;
    int err;
} InitResult;


typedef struct {
    CompressedSet *newClusters;
    Point *outlier;
    int numOutlier;
    int err;
} NewClusterResult;

double threshold = 2; /* soglia dell distanza che determina l'appartenenza di un punto ad un cluster */
double convergence_value = 10; /* valore di convergenza per la sitma del range in cui si trova il valore di k ottimo */
double alpha = 1; /* fattore moltiplicativo per la convergence_value per trovare il valore ottimo di k nel range trovato precedentemente */
double beta = 1; /* valore che determina se un compressed set è valido o meno in fase di validazione dei nuovi compressed set */

int computeAvaiableMemory();

int readPoint(FILE *fp, Point *points, int to_read, int o_size);

InitResult init(Cluster *discardSet, int numCluster, Point *points, int numPoint);

double euclideanDistance(Point point, double cluster_coords[DIMENSION]);

ClusteringResult clusteringAlongInitialCluster(Cluster *discardSet, int numCluster, Point *points, int numPoint);

ClusteringResult clustringAlongNewCluster(Cluster *clusters, int numCluster, Point *points, int numPoints);

GNCResult generateNewCompressed(Point *points, int numPoints, int startingId);

NewClusterResult kmeans(Point *points, int numPoint, int numCluster, int startingId, bool save);

double computeWcss(CompressedSet *clusters, int numCluster);

void create_datatypes(MPI_Datatype *cluster, MPI_Datatype *points, MPI_Datatype *clustered_points);

int main(int argc, char **argv) {

    int rankInWorld, numProcsInWorld;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankInWorld);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcsInWorld);

    MPI_Datatype ClusterDataType;
    MPI_Datatype PointDataType;
    MPI_Datatype ClusteredPointDataType;
    create_datatypes(&ClusterDataType, &PointDataType, &ClusteredPointDataType);

    MPI_Comm SLAVE_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, rankInWorld != 0, 0, &SLAVE_COMM);
    int rankInSlave, numPorcsSlave;
    MPI_Comm_rank(SLAVE_COMM, &rankInSlave);
    MPI_Comm_size(SLAVE_COMM, &numPorcsSlave);

    if (!rankInWorld) {
        if (argc < 2) {
            exit(WRONG_ARGUMENT);
            MPI_Abort(MPI_COMM_WORLD, WRONG_ARGUMENT);
            printf("Parametro richiesti: INPUT FILE");
        }

        char* INPUT_FILE = argv[1];

        /** MASTER **/

        int numInitialCluster = 3; /*numero di cluster "originali" */

        /* allocazione memoria per i discard set */

        Cluster *discardSetMaster = (Cluster *) malloc(numInitialCluster * sizeof(Cluster));
        if (!discardSetMaster) {
            printf("Allocazione memoria per i discard set fallita\n");
            MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
            exit(MEM_ALLOC_ERROR);
        }

        for (int i = 0; i < numInitialCluster; ++i) {
            discardSetMaster[i].id = i;
            discardSetMaster[i].size = 0;
            for (int j = 0; j < DIMENSION; ++j) {
                discardSetMaster[i].variance[j] = 0;
            }
        }

        /* Calcolo della memoria a disposizione */

        int avaiableMemoryMaster = computeAvaiableMemory();/* numero di punti che possono essere ospitati in memoria */
        if (avaiableMemoryMaster == 0) {
            printf("Memoria insufficiente\n");
            free(discardSetMaster), discardSetMaster = NULL;
            MPI_Abort(MPI_COMM_WORLD, NOT_ENOUGHT_MEMORY);
            exit(NOT_ENOUGHT_MEMORY);
        }
        int chunkSizeMaster = avaiableMemoryMaster;

        Point *points = (Point *) malloc(chunkSizeMaster * sizeof(Point));
        if (!points) {
            free(discardSetMaster), discardSetMaster = NULL;
            printf("Allocazione memoria per i points set fallita\n");
            MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
            exit(MEM_ALLOC_ERROR);
        }

        FILE *dataset = fopen(INPUT_FILE, READ_ONLY);
        if (!dataset) {
            free(discardSetMaster), discardSetMaster = NULL;
            free(points), points = NULL;
            printf("Impossibile aprire il file\n");
            MPI_Abort(MPI_COMM_WORLD, FILE_OPENING_ERROR);
            exit(FILE_OPENING_ERROR);
        }

        int numPoint = readPoint(dataset, points, avaiableMemoryMaster, 0);
        if (numPoint < 0) {
            free(discardSetMaster), discardSetMaster = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
            MPI_Abort(MPI_COMM_WORLD, numPoint);
            exit(numPoint);
        }

        /* Eelezione dei centroidi */

        InitResult iRes = init(discardSetMaster, numInitialCluster, points, numPoint);
        if (iRes.err < 0) {
            points = NULL;
            discardSetMaster = NULL;
            fclose(dataset), dataset = NULL;
            MPI_Abort(MPI_COMM_WORLD, iRes.err);
            exit(iRes.err);
        }
        points = iRes.points;
        numPoint = iRes.numPoints;
        iRes.points = NULL;

        /* comunica agli slave il modello */

        MPI_Bcast(discardSetMaster, numInitialCluster, ClusterDataType, MASTER_PROCESS, MPI_COMM_WORLD);
        free(discardSetMaster), discardSetMaster = NULL;

        /*
         * legge in anticipo il massimo numero di punti che può ospitare, in modo che quando arriverà la richiesta dello
         * slave, il master sarà già pronto
         */

        avaiableMemoryMaster = computeAvaiableMemory() - numPoint; /* -numPoint simula la memoria che si riempe */
        chunkSizeMaster = numPoint + avaiableMemoryMaster;

        Point *temp = (Point *) realloc(points, chunkSizeMaster * sizeof(Point));
        if (!temp) {
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            printf("allocazione memoria fallita\n");
            MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
            exit(MEM_ALLOC_ERROR);
        }
        points = temp, temp = NULL;

        numPoint = readPoint(dataset, points, avaiableMemoryMaster, numPoint);
        if (numPoint < 0) {
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
            MPI_Abort(MPI_COMM_WORLD, numPoint);
            exit(numPoint);
        } else if (numPoint < chunkSizeMaster) {
            temp = (Point *) realloc(points, numPoint * sizeof(Point));
            if (!temp) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            points = temp, temp = NULL;
            chunkSizeMaster = numPoint;
        }

        /* prima distribuzione del lavoro */

        for (int rank = 1; rank < numProcsInWorld; ++rank) {
            int numToSend;

            /* riceve quanti punti vuole lo slave */

            MPI_Recv(&numToSend, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Verifica di avere a disposizione numToSend punti
             * es.: lo slave ha a disposizione più memoria del master, allora il master invierà tutti e soli i punti
             * che possono essere contenuti nella sua memoria, altrimenti se numPoint >= numToSend allora il master ha
             * a disposizione un numero sufficiente di punti per soddisfare la richiesta dello slave e quindi invia
             * esattamente numToSend punti
             */

            if (numPoint < numToSend) numToSend = numPoint;

            /* Invia il giusto numero di punti allo slave */

            MPI_Send(points, numToSend, PointDataType, rank, 1, MPI_COMM_WORLD);
            numPoint -= numToSend;

            if (numPoint > 0) {
                /*
                 * allora non avrà inviato tutti i punti ma solo una parte, quindi i punti in eccesso vengono spostati
                 * indietro e points ridimensionato
                 */

                memmove(points, points + numToSend, numPoint * sizeof(Point));
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(dataset), dataset = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
            }

            /* carica il prossimo chunk di punti da inviare */

            avaiableMemoryMaster = computeAvaiableMemory() - numPoint;
            chunkSizeMaster = numPoint + avaiableMemoryMaster;

            temp = (Point *) realloc(points, chunkSizeMaster * sizeof(Point));
            if (!temp) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            points = temp, temp = NULL;

            numPoint = readPoint(dataset, points, avaiableMemoryMaster, numPoint);
            if (numPoint < 0) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
                MPI_Abort(MPI_COMM_WORLD, numPoint);
                exit(numPoint);
            } else if (numPoint < chunkSizeMaster) {
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(dataset), dataset = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
                chunkSizeMaster = numPoint;
            }
        }

        /* assegnazioni successive */

        FILE *outputFile = fopen(OUTPUT_FILE, APPEND_MODE);
        if (!outputFile) {
            free(discardSetMaster), discardSetMaster = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            printf("Impossibile aprire il file %s\n", OUTPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, FILE_OPENING_ERROR);
            exit(FILE_OPENING_ERROR);
        }

        FILE *outlierFile = fopen(OUTLIER_FILE, APPEND_MODE);
        if (!outputFile) {
            free(discardSetMaster), discardSetMaster = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            fclose(outputFile), outputFile = NULL;
            printf("Impossibile aprire il file %s\n", OUTPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
            exit(MEM_ALLOC_ERROR);
        }

        while (numPoint > 0) {
            MPI_Status statusCompressed;
            MPI_Status statusOutlier;
            int numClusteredPoint;
            int numOutlier;
            ClusteredPoint *recvClustered = NULL;
            Point *recvOutlier = NULL;
            int source;
            int numToSend;

            /* riceve il risultato del clustering */

            MPI_Probe(MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &statusCompressed);
            MPI_Get_count(&statusCompressed, ClusteredPointDataType, &numClusteredPoint);
            source = statusCompressed.MPI_SOURCE;
            recvClustered = (ClusteredPoint *) malloc(numClusteredPoint * sizeof(ClusteredPoint));

            if (!recvClustered) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvClustered, numClusteredPoint, ClusteredPointDataType, source, 2, MPI_COMM_WORLD,
                     &statusCompressed);

            MPI_Probe(source, 3, MPI_COMM_WORLD, &statusOutlier);
            MPI_Get_count(&statusOutlier, PointDataType, &numOutlier);

            recvOutlier = (Point *) malloc(numOutlier * sizeof(Point));
            if (!recvOutlier) {
                free(points), points = NULL;
                free(recvClustered), recvClustered = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvOutlier, numOutlier, PointDataType, source, 3, MPI_COMM_WORLD, &statusOutlier);

            /* riceve quanti punti vuole lo slave */

            MPI_Recv(&numToSend, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Invia il giusto numero di punti allo slave */

            if (numPoint < numToSend) numToSend = numPoint;
            MPI_Send(points, numToSend, PointDataType, source, 1, MPI_COMM_WORLD);
            numPoint -= numToSend;

            if (numPoint > 0) {
                /*
                 * allora non avrà inviato tutti i punti ma solo una parte, quindi i punti in eccesso vengono spostati
                 * indietro e points ridimensionato
                 */

                memmove(points, points + numToSend, numPoint * sizeof(Point));
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(dataset), dataset = NULL;
                    fclose(outputFile), outputFile = NULL;
                    fclose(outlierFile), outlierFile = NULL;
                    free(recvClustered), recvClustered = NULL;
                    free(recvOutlier), recvOutlier = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
            }

            /* scrive su file i clusteredPoint */

            for (int i = 0; i < numClusteredPoint; ++i)
                fprintf(outputFile, "%d,%d\n", recvClustered[i].idPoint, recvClustered[i].idCluster);

            /*scrive su file gli outlier */

            for (int i = 0; i < numOutlier; ++i) {
                fprintf(outlierFile, "%d", recvOutlier[i].id);
                for (int j = 0; j < DIMENSION; ++j) fprintf(outlierFile, ",%f", recvOutlier[i].coords[j]);
                fprintf(outlierFile, "\n");
            }

            free(recvClustered), recvClustered = NULL;
            free(recvOutlier), recvOutlier = NULL;

            /* carica il prossimo chunk di punti da inviare */

            avaiableMemoryMaster = computeAvaiableMemory() - numPoint;
            chunkSizeMaster = numPoint + avaiableMemoryMaster;

            temp = (Point *) realloc(points, chunkSizeMaster * sizeof(Point));
            if (!temp) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            points = temp, temp = NULL;

            numPoint = readPoint(dataset, points, avaiableMemoryMaster, numPoint);
            if (numPoint < 0) {
                free(points), points = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                fclose(dataset), dataset = NULL;
                printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
                MPI_Abort(MPI_COMM_WORLD, numPoint);
                exit(numPoint);
            } else if (numPoint < chunkSizeMaster && numPoint > 0) {
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(outputFile), outputFile = NULL;
                    fclose(outlierFile), outlierFile = NULL;
                    fclose(dataset), dataset = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
                chunkSizeMaster = numPoint;
            }
        }

        free(points), points = NULL;

        /* terminazione */

        for (int rank = 1; rank < numProcsInWorld; ++rank) {
            MPI_Status statusClustered;
            MPI_Status statusOutlier;
            int numClusteredPoint;
            int numOutlier;
            int toSend;
            ClusteredPoint *recvClustered;
            Point *recvOutlier;

            /* riceve il risultato del clustering */

            MPI_Probe(rank, 2, MPI_COMM_WORLD, &statusClustered);
            MPI_Get_count(&statusClustered, ClusteredPointDataType, &numClusteredPoint);
            recvClustered = (ClusteredPoint *) malloc(numClusteredPoint * sizeof(ClusteredPoint));
            if (!recvClustered) {
                fclose(dataset), dataset = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvClustered, numClusteredPoint, ClusteredPointDataType, rank, 2, MPI_COMM_WORLD,
                     &statusClustered);

            MPI_Probe(rank, 3, MPI_COMM_WORLD, &statusOutlier);
            rank = statusOutlier.MPI_SOURCE;
            MPI_Get_count(&statusOutlier, PointDataType, &numOutlier);
            recvOutlier = (Point *) malloc(numOutlier * sizeof(Point));
            if (!recvOutlier) {
                free(recvClustered), recvClustered = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvOutlier, numOutlier, PointDataType, rank, 3, MPI_COMM_WORLD, &statusOutlier);

            /* riceve quanti punti vuole lo slave */

            MPI_Recv(&toSend, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* invia il messaggio di terminazione */

            MPI_Send(points, 0, PointDataType, rank, 1, MPI_COMM_WORLD);

            /* scrive su file i clusteredPoint */

            for (int i = 0; i < numClusteredPoint; ++i)
                fprintf(outputFile, "%d,%d\n", recvClustered[i].idPoint, recvClustered[i].idCluster);

            /*scrive su file gli outlier */

            for (int i = 0; i < numOutlier; ++i) {
                fprintf(outlierFile, "%d", recvOutlier[i].id);
                for (int j = 0; j < DIMENSION; ++j) fprintf(outlierFile, ",%f", recvOutlier[i].coords[j]);
                fprintf(outlierFile, "\n");
            }

            free(recvClustered), recvClustered = NULL;
            free(recvOutlier), recvOutlier = NULL;

        }

        /* generazione nuovi cluster */

        fclose(outlierFile), outlierFile = NULL;
        fclose(dataset), dataset = NULL;

        dataset = fopen(OUTLIER_FILE, READ_ONLY);
        if (!dataset) {
            fclose(outputFile), outputFile = NULL;
            printf("Impossibili aprire il file %s\n", OUTLIER_FILE);
            MPI_Abort(MPI_COMM_WORLD, FILE_OPENING_ERROR);
            exit(FILE_OPENING_ERROR);
        }

        /*
        * legge in anticipo il massimo numero di punti che può ospitare, in modo che quando arriverà la richiesta dello
        * slave, il master sarà già pronto
        */

        avaiableMemoryMaster = computeAvaiableMemory() - numPoint; /* -numPoint simula la memoria che si riempe */
        chunkSizeMaster = numPoint + avaiableMemoryMaster;

        points = (Point *) malloc(chunkSizeMaster * sizeof(Point));
        if (!points) {
            fclose(dataset), dataset = NULL;
            printf("allocazione memoria fallita\n");
            MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
            exit(MEM_ALLOC_ERROR);
        }

        numPoint = readPoint(dataset, points, avaiableMemoryMaster, numPoint);
        if (numPoint < 0) {
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
            MPI_Abort(MPI_COMM_WORLD, numPoint);
            exit(numPoint);
        } else if (numPoint < chunkSizeMaster) {
            temp = (Point *) realloc(points, numPoint * sizeof(Point));
            if (!temp) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            points = temp, temp = NULL;
            chunkSizeMaster = numPoint;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        /* prima distribuzione del lavoro */

        for (int rank = 1; rank < numProcsInWorld; ++rank) {
            int numToSend;

            /* riceve quanti punti vuole lo slave */

            MPI_Recv(&numToSend, 1, MPI_INT, rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Verifica di avere a disposizione numToSend punti
             * es.: lo slave ha a disposizione più memoria del master, allora il master invierà tutti e soli i punti
             * che possono essere contenuti nella sua memoria, altrimenti se numPoint >= numToSend allora il master ha
             * a disposizione un numero sufficiente di punti per soddisfare la richiesta dello slave e quindi invia
             * esattamente numToSend punti
             */

            if (numPoint < numToSend) numToSend = numPoint;

            /* Invia il giusto numero di punti allo slave */

            MPI_Send(points, numToSend, PointDataType, rank, 5, MPI_COMM_WORLD);
            numPoint -= numToSend;

            if (numPoint > 0) {
                /*
                 * allora non avrà inviato tutti i punti ma solo una parte, quindi i punti in eccesso vengono spostati
                 * indietro e points ridimensionato
                 */

                memmove(points, points + numToSend, numPoint * sizeof(Point));
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(dataset), dataset = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
            }

            /* carica il prossimo chunk di punti da inviare */

            avaiableMemoryMaster = computeAvaiableMemory() - numPoint;
            chunkSizeMaster = numPoint + avaiableMemoryMaster;

            temp = (Point *) realloc(points, chunkSizeMaster * sizeof(Point));
            if (!temp) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            points = temp, temp = NULL;

            numPoint = readPoint(dataset, points, avaiableMemoryMaster, numPoint);
            if (numPoint < 0) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
                MPI_Abort(MPI_COMM_WORLD, numPoint);
                exit(numPoint);
            } else if (numPoint < chunkSizeMaster) {
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(dataset), dataset = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
                chunkSizeMaster = numPoint;
            }

        }

        /* assegnazioni successive */

        int procWNoMem = 0;

        while (numPoint > 0) {
            MPI_Status statusCompressed;
            int numClusteredPoint;
            ClusteredPoint *recvClustered = NULL;
            int source;
            int numToSend;

            /* riceve il risultato del clustering */

            MPI_Probe(MPI_ANY_SOURCE, 6, MPI_COMM_WORLD, &statusCompressed);
            MPI_Get_count(&statusCompressed, ClusteredPointDataType, &numClusteredPoint);
            source = statusCompressed.MPI_SOURCE;
            recvClustered = (ClusteredPoint *) malloc(numClusteredPoint * sizeof(ClusteredPoint));

            if (!recvClustered) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                fclose(outputFile), outputFile = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvClustered, numClusteredPoint, ClusteredPointDataType, source, 6, MPI_COMM_WORLD,
                     &statusCompressed);

            /* riceve quanti punti vuole lo slave */

            MPI_Recv(&numToSend, 1, MPI_INT, source, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (numToSend == 0) procWNoMem++;
            if (procWNoMem == numProcsInWorld - 1) {
                printf("Memoria insufficiente\n");
                fclose(dataset), dataset = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                MPI_Abort(MPI_COMM_WORLD, NOT_ENOUGHT_MEMORY);
                exit(NOT_ENOUGHT_MEMORY);

            }

            /* Invia il giusto numero di punti allo slave */


            if (numPoint < numToSend) numToSend = numPoint;
            MPI_Send(points, numToSend, PointDataType, source, 5, MPI_COMM_WORLD);
            numPoint -= numToSend;

            if (numPoint > 0) {
                /*
                 * allora non avrà inviato tutti i punti ma solo una parte, quindi i punti in eccesso vengono spostati
                 * indietro e points ridimensionato
                 */

                memmove(points, points + numToSend, numPoint * sizeof(Point));
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(dataset), dataset = NULL;
                    fclose(outputFile), outputFile = NULL;
                    free(recvClustered), recvClustered = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
            }

            /* scrive su file i clusteredPoint */

            for (int i = 0; i < numClusteredPoint; ++i)
                fprintf(outputFile, "%d,%d\n", recvClustered[i].idPoint, recvClustered[i].idCluster);
            free(recvClustered), recvClustered = NULL;

            /* carica il prossimo chunk di punti da inviare */

            avaiableMemoryMaster = computeAvaiableMemory() - numPoint;
            chunkSizeMaster = numPoint + avaiableMemoryMaster;

            temp = (Point *) realloc(points, chunkSizeMaster * sizeof(Point));
            if (!temp) {
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                fclose(outputFile), outputFile = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            points = temp, temp = NULL;

            numPoint = readPoint(dataset, points, avaiableMemoryMaster, numPoint);
            if (numPoint < 0) {
                free(points), points = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(dataset), dataset = NULL;
                printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
                MPI_Abort(MPI_COMM_WORLD, numPoint);
                exit(numPoint);
            } else if (numPoint < chunkSizeMaster && numPoint > 0) {
                temp = (Point *) realloc(points, numPoint * sizeof(Point));
                if (!temp) {
                    free(points), points = NULL;
                    fclose(dataset), dataset = NULL;
                    printf("allocazione memoria fallita\n");
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                points = temp, temp = NULL;
                chunkSizeMaster = numPoint;
            }
        }

        free(points), points = NULL;
        for (int rank = 1; rank < numProcsInWorld; ++rank) {

            MPI_Status statusClustered;
            MPI_Status statusOutlier;
            int numClusteredPoint;
            int numOutlier;
            int toSend;
            ClusteredPoint *recvClustered;
            Point *recvOutlier;

            /* riceve il risultato del clustering */

            MPI_Probe(rank, 6, MPI_COMM_WORLD, &statusClustered);
            MPI_Get_count(&statusClustered, ClusteredPointDataType, &numClusteredPoint);
            recvClustered = (ClusteredPoint *) malloc(numClusteredPoint * sizeof(ClusteredPoint));
            if (!recvClustered) {
                fclose(dataset), dataset = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvClustered, numClusteredPoint, ClusteredPointDataType, rank, 6, MPI_COMM_WORLD,
                     &statusClustered);

            /* riceve quanti punti vuole lo slave */

            MPI_Recv(&toSend, 1, MPI_INT, rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* invia il messaggio di terminazione */


            MPI_Send(points, 0, PointDataType, rank, 5, MPI_COMM_WORLD);

            /* riceve gli outlier */

            MPI_Probe(rank, 7, MPI_COMM_WORLD, &statusOutlier);
            rank = statusOutlier.MPI_SOURCE;
            MPI_Get_count(&statusOutlier, PointDataType, &numOutlier);
            recvOutlier = (Point *) malloc(numOutlier * sizeof(Point));
            if (!recvOutlier) {
                free(recvClustered), recvClustered = NULL;
                fclose(outputFile), outputFile = NULL;
                fclose(outlierFile), outlierFile = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvOutlier, numOutlier, PointDataType, rank, 7, MPI_COMM_WORLD, &statusOutlier);

            /* scrive su file i clusteredPoint */

            for (int i = 0; i < numClusteredPoint; ++i)
                fprintf(outputFile, "%d,%d\n", recvClustered[i].idPoint, recvClustered[i].idCluster);

            /*scrive su file gli outlier */

            for (int i = 0; i < numOutlier; ++i) fprintf(outputFile, "%d,OUTLIER\n", recvOutlier[i].id);

            free(recvClustered), recvClustered = NULL;
            free(recvOutlier), recvOutlier = NULL;

        }

        fclose(outputFile), outputFile = NULL;
        fclose(dataset), dataset = NULL;
        remove(OUTLIER_FILE);

    } else {

        /** SLAVE **/

        int numInitialCluster = 3; /*numero di cluster "originali" */

        /* allocazione memoria per i discard set */

        Cluster *discardSetSlave = (Cluster *) malloc(numInitialCluster * sizeof(Cluster));
        if (!discardSetSlave) {
            printf("Allocazione memoria per i discard set fallita\n");
            MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
            exit(MEM_ALLOC_ERROR);
        }

        for (int i = 0; i < numInitialCluster; ++i) {
            discardSetSlave[i].id = i;
            discardSetSlave[i].size = 0;
            for (int j = 0; j < DIMENSION; ++j) {
                discardSetSlave[i].variance[j] = 0;
            }
        }

        /* gli slave ricevono il modello del discard set */

        MPI_Bcast(discardSetSlave, numInitialCluster, ClusterDataType, MASTER_PROCESS, MPI_COMM_WORLD);

        /** PARTE DI RICEZIONE DEL LAVORO E COMPUTAZIONE **/

        while (1) {
            MPI_Status status;
            Point *recvPoint = NULL;
            int nRecvPoint;
            Cluster *recvDiscardSet = (Cluster *) malloc(numPorcsSlave * numInitialCluster * sizeof(Cluster));
            int numRecvDiscardSet = numPorcsSlave * numInitialCluster;

            /* Calcolo della memoria a disposizione */

            int chunkSizeSlave = computeAvaiableMemory(); /* numero di punti che possono essere ospitati in memoria */
            if (chunkSizeSlave == 0) {
                printf("Memoria insufficiente\n");
                free(discardSetSlave), discardSetSlave = NULL;
                MPI_Abort(MPI_COMM_WORLD, NOT_ENOUGHT_MEMORY);
                exit(NOT_ENOUGHT_MEMORY);
            }

            /* comunicano al master quanti punti possono ricevere */

            MPI_Send(&chunkSizeSlave, 1, MPI_INT, MASTER_PROCESS, 0, MPI_COMM_WORLD);

            /* ricevono il lavoro da eseguire */

            MPI_Probe(MASTER_PROCESS, 1, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PointDataType, &nRecvPoint);
            recvPoint = (Point *) malloc(nRecvPoint * sizeof(Point));
            if (!recvPoint) {
                printf("Impossibile allocare memoria\n");
                free(discardSetSlave), discardSetSlave = NULL;
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            MPI_Recv(recvPoint, nRecvPoint, PointDataType, MASTER_PROCESS, 1, MPI_COMM_WORLD, &status);
            if (nRecvPoint == 0) {
                free(recvPoint), recvPoint = NULL;
                break;
            }

            /* clustering */

            ClusteringResult cluRes = clusteringAlongInitialCluster(discardSetSlave, numInitialCluster, recvPoint,
                                                                    nRecvPoint);
            if (cluRes.err < 0) {
                recvPoint = NULL;
                discardSetSlave = NULL;
                MPI_Abort(MPI_COMM_WORLD, cluRes.err);
                exit(cluRes.err);
            }
            nRecvPoint = 0;
            recvPoint = NULL;

            /* comunicano al master il risultato del clustering */

            MPI_Send(cluRes.clusteredPoint, cluRes.numClusteredPoint, ClusteredPointDataType, MASTER_PROCESS, 2,
                     MPI_COMM_WORLD);
            MPI_Send(cluRes.outlier, cluRes.numOutlier, PointDataType, MASTER_PROCESS, 3, MPI_COMM_WORLD);
            free(cluRes.clusteredPoint), cluRes.clusteredPoint = NULL;
            free(cluRes.outlier), cluRes.outlier = NULL;

        }

        free(discardSetSlave), discardSetSlave = NULL;

        int numCluster = 0;
        Cluster *compressedSet = NULL;
        Point *pointSlave = NULL;
        int numPointSlave = 0;
        int iter = 0;

        MPI_Barrier(MPI_COMM_WORLD);

        while (1) {
            MPI_Status status;

            int nRecvPoint;

            /* Calcolo della memoria a disposizione */

            int chunkSizeSlave =
                    computeAvaiableMemory() - numPointSlave; /* -numPointSlave simula l'occupazione della memoria */

            /* comunicano al master quanti punti possono ricevere */

            MPI_Send(&chunkSizeSlave, 1, MPI_INT, MASTER_PROCESS, 4, MPI_COMM_WORLD);

            /* ricevono il lavoro da eseguire */

            MPI_Probe(MASTER_PROCESS, 5, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PointDataType, &nRecvPoint);

            Point *temp = (Point *) realloc(pointSlave, (nRecvPoint + numPointSlave) * sizeof(Point));
            if (!temp) {
                printf("Impossibile allocare memoria\n");
                if (pointSlave) free(pointSlave), pointSlave = NULL;
                if (compressedSet) free(compressedSet), compressedSet = NULL;
                MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                exit(MEM_ALLOC_ERROR);
            }
            pointSlave = temp, temp = NULL;
            MPI_Recv(pointSlave + numPointSlave, nRecvPoint, PointDataType, MASTER_PROCESS, 5, MPI_COMM_WORLD,
                     &status);
            numPointSlave += nRecvPoint;

            if (nRecvPoint == 0 && iter > 0) {

                MPI_Send(pointSlave, numPointSlave, PointDataType, MASTER_PROCESS, 7, MPI_COMM_WORLD);
                free(pointSlave), pointSlave = NULL;
                break;
            }

            /* clustering */

            ClusteringResult cluRes = clustringAlongNewCluster(compressedSet, numCluster, pointSlave,
                                                               numPointSlave);
            if (cluRes.err < 0) {
                compressedSet = NULL;
                pointSlave = NULL;
                MPI_Abort(MPI_COMM_WORLD, cluRes.err);
                exit(cluRes.err);
            }
            pointSlave = cluRes.outlier;
            numPointSlave = cluRes.numOutlier;
            cluRes.outlier = NULL;

            /* generazione nuovi cluster */

            GNCResult gncRes = generateNewCompressed(pointSlave, numPointSlave, numInitialCluster + numCluster);
            if (gncRes.err < 0) {
                compressedSet = NULL;
                pointSlave = NULL;
                MPI_Abort(MPI_COMM_WORLD, cluRes.err);
                exit(cluRes.err);
            }

            int numNewCluster = gncRes.numCluster;
            Cluster *newCluster = gncRes.cluster;
            gncRes.cluster = NULL;

            pointSlave = gncRes.outlier;
            gncRes.outlier = NULL;
            numPointSlave = gncRes.numOutlier;

            if (numNewCluster > 0) {

                /* i nuovi cluster vengono aggiunti a quelli già esistenti */

                Cluster *temp_c = (Cluster *) realloc(compressedSet,
                                                      (numCluster + numNewCluster) * sizeof(Cluster));
                if (!temp_c) {
                    printf("Impossibile allocare memoria\n");
                    free(compressedSet), compressedSet = NULL;
                    free(newCluster), newCluster = NULL;
                    free(pointSlave), pointSlave = NULL;
                    MPI_Abort(MPI_COMM_WORLD, MEM_ALLOC_ERROR);
                    exit(MEM_ALLOC_ERROR);
                }
                compressedSet = temp_c, temp_c = NULL;

                memcpy(compressedSet + numCluster, newCluster, numNewCluster * sizeof(Cluster));
                numCluster += numNewCluster;
                free(newCluster), newCluster = NULL;

            }

            /* invia il risultato al master */

            MPI_Send(gncRes.clusteredPoint, gncRes.numClusteredPoint, ClusteredPointDataType, MASTER_PROCESS, 6,
                     MPI_COMM_WORLD);
            free(gncRes.clusteredPoint), gncRes.clusteredPoint = NULL;
            iter++;

        }

        free(compressedSet), compressedSet = NULL;
        free(pointSlave), pointSlave = NULL;

        MPI_Comm_free(&SLAVE_COMM);
    }

    MPI_Type_free(&ClusterDataType);
    MPI_Type_free(&PointDataType);
    MPI_Type_free(&ClusteredPointDataType);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();


    return 0;


}

void create_datatypes(MPI_Datatype *cluster, MPI_Datatype *points, MPI_Datatype *clustered_points) {
    int cluster_size[5] = {1, 1, DIMENSION, DIMENSION, DIMENSION};
    int points_size[2] = {1, DIMENSION};
    int clustered_size[3] = {1, 1};

    MPI_Datatype cluster_type[5] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype points_type[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Datatype clustered_type[3] = {MPI_INT, MPI_INT};

    MPI_Aint cluster_offset[5] = {offsetof(Cluster, id), offsetof(Cluster, size), offsetof(Cluster, variance),
                                  offsetof(Cluster, coord), offsetof(Cluster, squared_coord)};
    MPI_Aint points_offset[2] = {offsetof(Point, id), offsetof(Point, coords)};
    MPI_Aint clustered_offset[3] = {offsetof(ClusteredPoint, idPoint), offsetof(ClusteredPoint, idCluster)};

    MPI_Type_create_struct(5, cluster_size, cluster_offset, cluster_type, cluster);
    MPI_Type_create_struct(2, points_size, points_offset, points_type, points);
    MPI_Type_create_struct(2, clustered_size, clustered_offset, clustered_type, clustered_points);

    MPI_Type_commit(cluster);
    MPI_Type_commit(points);
    MPI_Type_commit(clustered_points);
}

int computeAvaiableMemory() {
    return 35000;
}

int readPoint(FILE *fp, Point *points, int to_read, int o_size) {
    int read = 0;

    while (read < to_read) {
        size_t len = 0;
        char *line = NULL;

        if (getline(&line, &len, fp) == -1) {
            if (feof(fp)) return read + o_size;
            else return FILE_READING_ERROR;
        }

        char *token = strtok(line, ","); /* tokenize della riga che contiene un punto letta dal file*/
        if (!token) return FILE_READING_ERROR;

        points[read + o_size].id = atoi(token); /* Lettura id del punto, ovvero il primo token */

        /* lettura coordinate */

        for (int j = 0; j < DIMENSION; j++) {
            token = strtok(NULL, ",");
            if (!token) return FILE_READING_ERROR;
            points[read + o_size].coords[j] = atof(token);
        }

        free(line), line = NULL, read++;
    }
    return read + o_size;
}

InitResult init(Cluster *discardSet, int numCluster, Point *points, int numPoint) {
    /* ogni volta che un punto viene eletto come centro viene immediatamente assegnato alla memoria scondaria */
    InitResult res;
    res.points = NULL;
    res.err = 0;

    FILE *outputFile = fopen(OUTPUT_FILE, APPEND_MODE);
    if (!outputFile) {
        printf("Non è stato possibile aprire o creare il file %s\n", OUTPUT_FILE);
        free(points), points = NULL;
        free(discardSet), discardSet = NULL;
        res.err = FILE_OPENING_ERROR;
        return res;
    }
    fprintf(outputFile, "id,cluster\n");

    /* elezione dei centri */

    for (int cluster_idx = 0; cluster_idx < numCluster; ++cluster_idx) {
        int pointIndex = -1; /* indice del punto da assegnare all'i-esimo cluster*/

        if (!cluster_idx) srand(time(NULL)), pointIndex = rand() % numPoint; //generazione numero casuale tra 0 e numPoint - 1 per il primo cluster
        else { /* cluster successivi */
            for (int j = 0; j < numPoint; ++j) {
                int check = 1;

                /* se il j-esimo punto è sufficientemente lontano da dai centri dei cluster già inizializzati
                 * allora viene scelto come centro per inizializzare il cluster di indice cluser_idx*/

                for (int k = 0; k < cluster_idx; ++k) {
                    double distance = euclideanDistance(points[j], discardSet[k].coord);
                    if (distance < threshold) {
                        check = 0;
                        break;
                    }
                }

                if (check) {
                    pointIndex = j;
                    break;
                }
            }
        }

        if (pointIndex == -1) {

            /* Non è stato possibile inizializzare "numCluster" cluster */

            free(points), points = NULL;
            free(discardSet), discardSet = NULL;
            fclose(outputFile), outputFile = NULL;
            res.err = INIT_ERROR;
            return res;
        }

        /* aggiornamento statistiche del cluster */

        discardSet[cluster_idx].size++;
        for (int j = 0; j < DIMENSION; j++) {
            discardSet[cluster_idx].coord[j] = points[pointIndex].coords[j];
            discardSet[cluster_idx].squared_coord[j] = pow(points[pointIndex].coords[j], 2);
        }

        /* Scrittura su file del punto */

        fprintf(outputFile, "%d,%d\n", points[pointIndex].id,
                discardSet[cluster_idx].id); /* scrittura del punto in memoria secondaria */

        /* Il punto viene rimosso dalla memoria */

        memmove(points + pointIndex, points + pointIndex + 1, (numPoint - pointIndex - 1) * sizeof(Point));
        numPoint--;

        Point *temp = (Point *) realloc(points, numPoint * sizeof(Point));
        if (!temp) {
            free(points), points = NULL;
            free(discardSet), discardSet = NULL;
            fclose(outputFile), outputFile = NULL;
            res.err = MEM_ALLOC_ERROR;
            return res;
        }
        points = temp, temp = NULL;

    }

    fclose(outputFile), outputFile = NULL;
    res.points = points;
    res.numPoints = numPoint;
    return res;
}

double euclideanDistance(Point point, double cluster_coords[DIMENSION]) {
    double distance_squared = 0;
    for (int i = 0; i < DIMENSION; i++)
        distance_squared += pow((point.coords[i] - cluster_coords[i]), 2);
    return sqrt(distance_squared);
}

ClusteringResult clusteringAlongInitialCluster(Cluster *discardSet, int numCluster, Point *points, int numPoint) {
    ClusteringResult res;
    res.clusteredPoint = NULL;
    res.outlier = NULL;
    res.numClusteredPoint = 0;
    res.numOutlier = 0;
    res.err = 0;

    ClusteredPoint *clusteredPoint = NULL;
    Point *outlier = NULL;
    int numClusteredPoint = 0;
    int numOutlier = 0;

    for (int pointindex = 0; pointindex < numPoint; ++pointindex) { /* index: indice del punto */
        double min_distance = INFINITY;
        int clusterIdx = -1; /* indice del cluster scelto */

        /* valutazione dell'appartenenza dell'i-esimo punto all j-esimo discard set */

        for (int j = 0; j < numCluster; ++j) {

            /* disatnza del punto i dal cluster j */

            double distance = euclideanDistance(points[pointindex], discardSet[j].coord);

            /* se viene trovato un candidato mogliore */

            if (distance < min_distance && distance < threshold) {
                min_distance = distance;
                clusterIdx = j;
            }
        }

        if (clusterIdx != -1) { /* punto assegnato a un cluster */

            /* aggiornamento statistiche cluster */

            int n = ++discardSet[clusterIdx].size; /* numero di punti nel cluster prima di assegnare il nuovo punto */

            /* aggiornamento delle statistiche del cluster */

            for (int j = 0; j < DIMENSION; ++j) {

                /* sum[i]/n */
                discardSet[clusterIdx].coord[j] =
                        ((discardSet[clusterIdx].coord[j] * n) + points[pointindex].coords[j]) / (n + 1);

                /* sumq[i]/n */
                discardSet[clusterIdx].squared_coord[j] =
                        ((discardSet[clusterIdx].squared_coord[j] * n) + pow(points[pointindex].coords[j], 2)) /
                        (n + 1);

                /* varianza: la varianza è calcolata come sumq[i]/n - (sum[i]/n)^2 */
                discardSet[clusterIdx].variance[j] =
                        discardSet[clusterIdx].squared_coord[j] - pow(discardSet[clusterIdx].coord[j], 2);
            }

            /* il punto viene inserito in clusteredPoint */

            numClusteredPoint++;
            ClusteredPoint *temp = (ClusteredPoint *) realloc(clusteredPoint,
                                                              numClusteredPoint * sizeof(ClusteredPoint));
            if (!temp) {
                if (clusteredPoint) free(clusteredPoint), clusteredPoint = NULL;
                if (outlier) free(outlier), outlier = NULL;
                free(points), points = NULL;
                free(discardSet), discardSet = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            clusteredPoint = temp, temp = NULL;

            clusteredPoint[numClusteredPoint - 1].idPoint = points[pointindex].id;
            clusteredPoint[numClusteredPoint - 1].idCluster = clusterIdx;

        } else {
            /* il punto viene inserito in outlier */

            numOutlier++;
            Point *temp = (Point *) realloc(outlier, numOutlier * sizeof(Point));
            if (!temp) {
                if (clusteredPoint) free(clusteredPoint), clusteredPoint = NULL;
                if (outlier) free(outlier), outlier = NULL;
                free(points), points = NULL;
                free(discardSet), discardSet = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            outlier = temp, temp = NULL;

            memcpy(outlier + numOutlier - 1, points + pointindex, sizeof(Point));
        }

        /* Il punto viene rimosso dalla memoria */

        memmove(points + pointindex, points + pointindex + 1, (numPoint - pointindex - 1) * sizeof(Point));
        numPoint--;

        Point *temp = (Point *) realloc(points, numPoint * sizeof(Point));
        if (!temp) {
            printf("Allocazione della memoria fallita\n");
            free(points), points = NULL;
            free(discardSet), discardSet = NULL;
            if (clusteredPoint) free(clusteredPoint), clusteredPoint = NULL;
            if (outlier) free(outlier), outlier = NULL;
            res.err = MEM_ALLOC_ERROR;
            return res;
        }
        points = temp, temp = NULL;

        pointindex--; /* altrimenti si saltano punti */
    }

    /* alla fine di questa fase point sarà vuoto */

    free(points);

    res.clusteredPoint = clusteredPoint;
    clusteredPoint = NULL;
    res.numClusteredPoint = numClusteredPoint;
    res.outlier = outlier;
    outlier = NULL;
    res.numOutlier = numOutlier;

    return res;

}

ClusteringResult clustringAlongNewCluster(Cluster *clusters, int numCluster, Point *points, int numPoints) {

    ClusteringResult res;
    res.clusteredPoint = NULL;
    res.outlier = NULL;
    res.numClusteredPoint = 0;
    res.numOutlier = 0;
    res.err = 0;

    if (numCluster == 0) {
        res.outlier = points;
        res.numOutlier = numPoints;
        return res;
    }

    ClusteredPoint *clusteredPoint = NULL;
    Point *outlier = NULL;
    int numClusteredPoint = 0;
    int numOutlier = 0;

    for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex) { /* index indice del punto */
        double min_distance = INFINITY;
        int clusterIdx = -1; /* indice del cluster scelto */

        /* valutazione dell'appartenenza del i-esimo punto all j-esimo compressed set */

        for (int j = 0; j < numCluster; ++j) {

            /* disatnza del punto i dal cluster j */

            double distance = euclideanDistance(points[pointIndex], clusters[j].coord);

            /* se viene trovato un candidato mogliore */

            if (distance < min_distance && distance < threshold) min_distance = distance, clusterIdx = j;
        }

        if (clusterIdx != -1) { /* il punto appartiene a un cluster */

            /* aggiornamento statistiche cluster */

            int n = ++clusters[clusterIdx].size; /* numero di punti nel cluster prima di assegnare il nuovo punto */

            /* aggiornamento delle coordinate del centro */

            for (int j = 0; j < DIMENSION; ++j) {

                /* sum[i]/n */

                clusters[clusterIdx].coord[j] =
                        ((clusters[clusterIdx].coord[j] * n) + points[pointIndex].coords[j]) / (n + 1);

                /* sumq[i]/n */

                clusters[clusterIdx].squared_coord[j] =
                        ((clusters[clusterIdx].squared_coord[j] * n) + pow(points[pointIndex].coords[j], 2)) / (n + 1);

                /* varianza: la varianza è calcolata come sumq[i]/n - (sum[i]/n)^2 */

                clusters[clusterIdx].variance[j] =
                        clusters[clusterIdx].squared_coord[j] - pow(clusters[clusterIdx].coord[j], 2);
            }

            /* Il punto viene aggiunto a clustered point */

            numClusteredPoint++;
            ClusteredPoint *temp = (ClusteredPoint *) realloc(clusteredPoint,
                                                              numClusteredPoint * sizeof(ClusteredPoint));
            if (!temp) {
                if (clusteredPoint) free(clusteredPoint), clusteredPoint = NULL;
                if (outlier) free(outlier), outlier = NULL;
                free(points), points = NULL;
                free(clusters), clusters = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            clusteredPoint = temp, temp = NULL;

            clusteredPoint[numClusteredPoint - 1].idPoint = points[pointIndex].id;
            clusteredPoint[numClusteredPoint - 1].idCluster = clusters[clusterIdx].id;

        } else {
            /* il punto viene inserito tra gli outlier */

            numOutlier++;
            Point *temp = (Point *) realloc(outlier, numOutlier * sizeof(Point));
            if (!temp) {
                printf("Errore nell'allocazione della memoria\n");
                if (outlier)
                    free(outlier), outlier = NULL; /* qunado arriva qui potrebbe non aver ancora inizializzato outlier */
                free(points), points = NULL;
                free(clusters), clusters = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            outlier = temp, temp = NULL;

            outlier[numOutlier - 1] = points[pointIndex];

        }

        /* eliminazione del punto da point */

        memmove(points + pointIndex, points + pointIndex + 1, (numPoints - pointIndex - 1) * sizeof(Point));
        numPoints--;

        Point *temp = (Point *) realloc(points, numPoints * sizeof(Point));
        if (!temp) {
            printf("Allocazione della memoria fallita\n");
            free(points), points = NULL;
            free(clusters), clusters = NULL;
            if (clusteredPoint) free(clusteredPoint), clusteredPoint = NULL;
            if (outlier)
                free(outlier), outlier = NULL;  /* qunado arriva qui potrebbe non aver ancora inizializzato outlier */
            res.err = MEM_ALLOC_ERROR;
            return res;
        }
        points = temp, temp = NULL;

        pointIndex--; /* altrimenti si saltano punti */
    }

    free(points), points = NULL;
    res.outlier = outlier;
    outlier = NULL;
    res.numOutlier = numOutlier;
    return res;
}

GNCResult generateNewCompressed(Point *points, int numPoints, int startingId) {

    GNCResult res;
    res.cluster = NULL;
    res.outlier = NULL;
    res.clusteredPoint = NULL;
    res.numCluster = 0;
    res.outlier = 0;
    res.numClusteredPoint = 0;
    res.err = 0;

    bool convergence = false;
    if (numPoints == 0) return res;

    int k, max_k = (int) pow(2, (int) log2(numPoints));
    double wcss_prev = 0, wcss = INFINITY;

    /* ricerca del range [k/2, k] in cui si trova il valore ottimo di k, la ricerca si ferma quando wcss converge */

    for (k = 1; k < max_k; k *= 2) {
        wcss_prev = wcss;

        /* esecuzione del clustring con k cluster */

        NewClusterResult ncRes = kmeans(points, numPoints, k, startingId, false);
        if (ncRes.err == NOT_ABLE_TO_CLUSTER) {
            res.outlier = points;
            res.numOutlier = numPoints;
            res.numCluster = 0;
            return res;
        } else if (ncRes.err < 0) {
            points = NULL;
            res.err = ncRes.err;
            return res;
        }

        wcss = computeWcss(ncRes.newClusters, k);
        for (int i = 0; i < k; ++i) free(ncRes.newClusters[i].points), ncRes.newClusters[i].points = NULL;
        free(ncRes.newClusters), ncRes.newClusters = NULL;
        free(ncRes.outlier), ncRes.outlier = NULL;

        if (fabs(wcss_prev - wcss) < convergence_value) {
            convergence = true;
            break;
        }
    }

    if (!convergence) {
        res.outlier = points;
        res.numOutlier = numPoints;
        res.numCluster = 0;

        return res;
    }

    /* dopo aver individuato il range [k/2, k], si esegue un'altra binary search per trovare l'esatto valore ottimo di k
     * anche in questo caso la ricerca si ferma quando la wcss converge */

    int x = k / 2, y = k;
    wcss_prev = wcss;
    while ((y - x) > 1) {
        k = (x + y) / 2; /* nuovo valore di k */

        NewClusterResult ncRes = kmeans(points, numPoints, k, startingId, false);
        if (ncRes.err < 0) {
            points = NULL;
            res.err = ncRes.err;
            return res;
        }

        wcss = computeWcss(ncRes.newClusters, k);
        for (int i = 0; i < k; ++i) free(ncRes.newClusters[i].points), ncRes.newClusters[i].points = NULL;
        free(ncRes.newClusters), ncRes.newClusters = NULL;
        free(ncRes.outlier), ncRes.outlier = NULL;

        if (fabs(wcss_prev - wcss) <= alpha * convergence_value) y = k, wcss_prev = wcss;
        else x = k;
    }

    k = y; /* valore ottimo di k */

    /* clustering con il valore ottimo per il numero di cluter */

    NewClusterResult ncRes = kmeans(points, numPoints, k, startingId, true);
    if (ncRes.err < 0) {
        points = NULL;
        res.err = ncRes.err;
        return res;
    }

    points = NULL;
    CompressedSet *clusters = ncRes.newClusters;
    ncRes.newClusters = NULL;
    Point *outlier = ncRes.outlier;
    ncRes.outlier = NULL;
    int numOutlier = ncRes.numOutlier;
    ClusteredPoint *clusteredPoint = NULL;
    int numClusteredPoint = 0;

    /*validazione dei cluster */

    for (int clusterIndex = 0; clusterIndex < k; ++clusterIndex) {
        int valid = 1;

        /* almeno due punti nel cluster */

        if (clusters[clusterIndex].size > 2) {
            for (int j = 0; j < DIMENSION && valid; ++j)

                /* varianza sulla dimensione j-esima entro la soglia */

                if (clusters[clusterIndex].variance[j] > beta)
                    valid = 0;
        } else valid = 0;

        if (!valid) {

            /*
             * Cluster non valido:  il cluster viene rimosso.
             * I punti appartenenti al cluster vengono salvati in outlier
             */

            Point *temp = (Point *) realloc(outlier, (numOutlier + clusters[clusterIndex].size) * sizeof(Point));
            if (!temp) {
                printf("Impossibile allocare la memoria\n");
                for (int j = 0; j < k; ++j) free(clusters[j].points), clusters[j].points = NULL;
                free(clusters), clusters = NULL;
                if (outlier) free(outlier), outlier = NULL; /* arrivato qui outlier potrebbe essere vuoto */
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            outlier = temp, temp = NULL;

            Point *temp_p = clusters[clusterIndex].points;
            memcpy(outlier + numOutlier, temp_p, clusters[clusterIndex].size * sizeof(Point));
            //for (int j = 0; j < clusters[clusterIndex].size; ++j) outlier[numOutlier + j] = clusters[clusterIndex].points[j];

            numOutlier += clusters[clusterIndex].size;
            free(clusters[clusterIndex].points), clusters[clusterIndex].points = NULL;

            /* il cluster viene eliminato */

            memmove(clusters + clusterIndex, clusters + clusterIndex + 1,
                    (k - clusterIndex - 1) * sizeof(CompressedSet));
            k--;

            CompressedSet *temp_cs = (CompressedSet *) realloc(clusters, k * sizeof(CompressedSet));
            if (!temp_cs) {
                printf("Impossibile allocare la memoria\n");
                for (int j = 0; j < k; ++j) free(clusters[j].points), clusters[j].points = NULL;
                free(clusters), clusters = NULL;
                free(outlier), outlier = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            clusters = temp_cs, temp_cs = NULL;

            clusterIndex--; /* altrimenti vengono saltati alcuni cluster */
        } else {

            /* cambio l'id del cluster */
            clusters[clusterIndex].id = startingId;
            startingId++;

            /* il cluster è valido: i punti vengono salvati in clusteredPoint */

            int nToAdd = clusters[clusterIndex].size;
            numClusteredPoint += nToAdd;
            ClusteredPoint *temp = (ClusteredPoint *) realloc(clusteredPoint,
                                                              numClusteredPoint * sizeof(ClusteredPoint));
            if (!temp) {
                printf("Impossibile allocare la memoria\n");
                for (int j = 0; j < k; ++j) free(clusters[j].points), clusters[j].points = NULL;
                free(clusters), clusters = NULL;
                free(outlier), outlier = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            clusteredPoint = temp, temp = NULL;


            for (int i = 0; i < clusters[clusterIndex].size; ++i) {
                clusteredPoint[numClusteredPoint - nToAdd + i].idPoint = clusters[clusterIndex].points[i].id;
                clusteredPoint[numClusteredPoint - nToAdd + i].idCluster = clusters[clusterIndex].id;
            }

            free(clusters[clusterIndex].points), clusters[clusterIndex].points = NULL;
        }
    }

    Cluster *toSave = (Cluster *) malloc(k * sizeof(Cluster));
    if (!toSave) {
        printf("Impossibile allocare memoria\n");
        for (int j = 0; j < k; ++j) free(clusters[j].points), clusters[j].points = NULL;
        free(clusters), clusters = NULL;
        if (outlier) free(outlier), outlier = NULL;
        res.err = MEM_ALLOC_ERROR;
        return res;
    }

    for (int clusterIndex = 0; clusterIndex < k; ++clusterIndex) {
        toSave[clusterIndex].id = clusters[clusterIndex].id;
        for (int j = 0; j < DIMENSION; ++j) {
            toSave[clusterIndex].coord[j] = clusters[clusterIndex].coord[j];
            toSave[clusterIndex].squared_coord[j] = clusters[clusterIndex].squared_coord[j];
            toSave[clusterIndex].variance[j] = clusters[clusterIndex].variance[j];
        }
    }

    res.cluster = toSave;
    toSave = NULL;
    res.numCluster = k;

    res.outlier = outlier;
    outlier = NULL;
    res.numOutlier = numOutlier;

    res.clusteredPoint = clusteredPoint;
    clusteredPoint = NULL;
    res.numClusteredPoint = numClusteredPoint;

    return res;
}

double computeWcss(CompressedSet *clusters, int numCluster) {
    double wcss = 0;
    for (int clusterIndex = 0; clusterIndex < numCluster; clusterIndex++) {
        for (int pointIndex = 0; pointIndex < clusters[clusterIndex].size; pointIndex++) {
            double distance = euclideanDistance(clusters[clusterIndex].points[pointIndex],
                                                clusters[clusterIndex].coord);
            wcss += pow(distance, 2);
        }
    }
    return wcss / numCluster;
}

NewClusterResult kmeans(Point *points, int numPoint, int numCluster, int startingId, bool save) {
    NewClusterResult res;
    res.newClusters = NULL;
    res.outlier = NULL;
    res.numOutlier = 0;
    res.err = 0;

    CompressedSet *compressedSet = (CompressedSet *) malloc(numCluster * sizeof(CompressedSet));
    if (!compressedSet) {
        printf("Impossibile allocare memoria\n");
        free(points), points = NULL;
        res.err = 0;
        return res;
    }

    for (int clusterIndex = 0; clusterIndex < numCluster; ++clusterIndex) {
        compressedSet[clusterIndex].id = startingId + clusterIndex;
        compressedSet[clusterIndex].size = 0;
        for (int j = 0; j < DIMENSION; ++j) {
            compressedSet[clusterIndex].variance[j] = 0;
        }
    }

    /* inizializzazione cluster */

    for (int clusterIdx = 0; clusterIdx < numCluster; ++clusterIdx) {
        int pointIdx = -1; /* indice del punto da assegnare all'i-esimo cluster*/

        if (!clusterIdx) srand(time(NULL)), pointIdx = rand() % numPoint; //generazione numero casuale tra 0 e numPoint - 1 per il primo cluster;
        else { /* cluster successivi */
            for (int j = 0; j < numPoint; ++j) {
                int check = 1;

                /* se il j-esimo punto è sufficientemente lontano da dai centri dei cluster già inizializzati
                 * allora viene scelto come centro per inizializzare il cluster di indice cluser_idx*/

                for (int k = 0; k < clusterIdx; ++k) {
                    double distance = euclideanDistance(points[j], compressedSet[k].coord);
                    if (distance < threshold) {
                        check = 0;
                        break;
                    }
                }

                if (check) {
                    pointIdx = j;
                    break;
                }
            }
        }

        if (pointIdx == -1) {

            /* Non è stato possibile inizializzare "numCluster" cluster */

            for (int i = 0; i < clusterIdx; ++i) free(compressedSet[i].points), compressedSet[i].points = NULL;
            free(compressedSet), compressedSet = NULL;
            res.err = NOT_ABLE_TO_CLUSTER;
            return res;
        }

        /* aggiornamento statistiche del cluster */

        compressedSet[clusterIdx].size++;
        for (int j = 0; j < DIMENSION; j++) {
            compressedSet[clusterIdx].coord[j] = points[pointIdx].coords[j];
            compressedSet[clusterIdx].squared_coord[j] = pow(points[pointIdx].coords[j], 2);
        }

        /* Assegnazione punto */
        compressedSet[clusterIdx].points = (Point *) malloc(sizeof(Point));
        if (!compressedSet[clusterIdx].points) {
            printf("Impossibile allocare memoria\n");
            for (int i = 0; i < clusterIdx; ++i) free(compressedSet[i].points), compressedSet[i].points = NULL;
            free(compressedSet), compressedSet = NULL;
            free(points), points = NULL;
            res.err = MEM_ALLOC_ERROR;
            return res;
        }
        compressedSet[clusterIdx].points[0] = points[pointIdx];

        if (save) {
            /* Il punto viene rimosso da points */

            memmove(points + pointIdx, points + pointIdx + 1, (numPoint - pointIdx - 1) * sizeof(Point));
            numPoint--;

            Point *temp = (Point *) realloc(points, numPoint * sizeof(Point));
            if (!temp) {
                for (int i = 0; i < clusterIdx + 1; ++i) free(compressedSet[i].points), compressedSet[i].points = NULL;
                free(compressedSet), compressedSet = NULL;
                free(points), points = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            points = temp, temp = NULL;
        } else {
            /* il punto viene solo spostato in coda */

            numPoint--;
            Point temp = points[pointIdx];
            points[pointIdx] = points[numPoint];
            points[numPoint] = temp;
        }

    }

    /* clustering */
    Point *outlier = NULL;
    int numOutlier = 0;

    for (int pointIndex = 0; pointIndex < numPoint; ++pointIndex) { /* index indice del punto */
        double min_distance = INFINITY;
        int clusterIdx = -1; /* indice del cluster scelto */

        /* valutazione dell'appartenenza del i-esimo punto all j-esimo compressed set */

        for (int j = 0; j < numCluster; ++j) {

            /* disatnza del punto i dal cluster j */

            double distance = euclideanDistance(points[pointIndex], compressedSet[j].coord);

            /* se viene trovato un candidato mogliore */

            if (distance < min_distance && distance < threshold) min_distance = distance, clusterIdx = j;
        }

        if (clusterIdx != -1) { /* il punto appartiene a un cluster */

            /* aggiornamento statistiche cluster */

            int n = ++compressedSet[clusterIdx].size; /* numero di punti nel cluster prima di assegnare il nuovo punto */

            /* aggiornamento delle coordinate del centro */

            for (int j = 0; j < DIMENSION; ++j) {

                /* sum[i]/n */

                compressedSet[clusterIdx].coord[j] =
                        ((compressedSet[clusterIdx].coord[j] * n) + points[pointIndex].coords[j]) / (n + 1);

                /* sumq[i]/n */

                compressedSet[clusterIdx].squared_coord[j] =
                        ((compressedSet[clusterIdx].squared_coord[j] * n) + pow(points[pointIndex].coords[j], 2)) /
                        (n + 1);

                /* varianza: la varianza è calcolata come sumq[i]/n - (sum[i]/n)^2 */

                compressedSet[clusterIdx].variance[j] =
                        compressedSet[clusterIdx].squared_coord[j] - pow(compressedSet[clusterIdx].coord[j], 2);
            }

            /* Assegnazione punto */

            Point *temp = (Point *) realloc(compressedSet[clusterIdx].points, n * sizeof(Point));
            if (!temp) {
                for (int i = 0; i < numCluster; ++i) free(compressedSet[i].points), compressedSet[i].points = NULL;
                free(compressedSet), compressedSet = NULL;
                if (outlier) free(outlier), outlier = NULL;
                free(points), points = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            compressedSet[clusterIdx].points = temp, temp = NULL;
            compressedSet[clusterIdx].points[n - 1] = points[pointIndex];

        } else if (save) {
            /* il punto viene inserito tra gli outlier */

            numOutlier++;
            Point *temp = (Point *) realloc(outlier, numOutlier * sizeof(Point));
            if (!temp) {
                printf("impossibile allocare memoria\n");
                for (int i = 0; i < numCluster; ++i) free(compressedSet[i].points), compressedSet[i].points = NULL;
                if (outlier) free(outlier), outlier = NULL;
                free(compressedSet), compressedSet = NULL;
                free(points), points = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            outlier = temp, temp = NULL;
            outlier[numOutlier - 1] = points[pointIndex];
        }

        if (save) {
            /* eliminazione del punto da point */

            memmove(points + pointIndex, points + pointIndex + 1, (numPoint - pointIndex - 1) * sizeof(Point));
            numPoint--;

            Point *temp = (Point *) realloc(points, numPoint * sizeof(Point));
            if (!temp) {
                printf("impossibile allocare memoria\n");
                for (int i = 0; i < numCluster; ++i) free(compressedSet[i].points), compressedSet[i].points = NULL;
                free(compressedSet), compressedSet = NULL;
                if (outlier) free(outlier), outlier = NULL;
                free(points), points = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            points = temp, temp = NULL;

            pointIndex--; /* altrimenti si saltano punti */
        }
    }
    if (save) free(points), points = NULL;

    res.newClusters = compressedSet;
    compressedSet = NULL;
    res.outlier = outlier;
    outlier = NULL;
    res.numOutlier = numOutlier;
    return res;
}


#pragma clang diagnostic pop
