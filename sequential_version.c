#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"
#pragma ide diagnostic ignored "cert-msc51-cpp"
#pragma ide diagnostic ignored "readability-non-const-parameter"
#pragma ide diagnostic ignored "misc-no-recursion"
#pragma ide diagnostic ignored "cert-msc50-cpp"
#pragma ide diagnostic ignored "cert-err34-c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc/malloc.h>
#include <math.h>
#include <sys/time.h>
#include <stdbool.h>

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
    Point *outlier;
    int numOutlier;
    int err;
} ClusteringResult;

typedef struct {
    Cluster *cluster;
    Point *outlier;
    int numCluster;
    int numOutlier;
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
double beta = 2; /* valore che determina se un compressed set è valido o meno in fase di validazione dei nuovi compressed set */

int computeAvaiableMemory();

int readPoint(FILE *fp, Point *points, int to_read, int o_size);

InitResult init(Cluster *discardSet, int numCluster, Point *points, int numPoint);

double euclideanDistance(Point point, double cluster_coords[DIMENSION]);

int clusteringAlongInitialCluster(Cluster *discardSet, int numCluster, Point *points, int numPoint);

ClusteringResult clustringAlongNewCluster(Cluster *clusters, int numCluster, Point *points, int numPoints);

GNCResult generateNewCompressed(Point *points, int numPoints, int startingId);

NewClusterResult kmeans(Point *points, int numPoint, int numCluster, int startingId, bool save);

double computeWcss(CompressedSet *clusters, int numCluster);

int main(int argc, char **argv) {

    if (argc < 2) {
        exit(WRONG_ARGUMENT);
        printf("Parametro richiesti: INPUT FILE");
    }
    char* INPUT_FILE = argv[1];

    int numInitialCluster = 3; /*numero di cluster "originali" */

    /* allocazione memoria per i discard set */

    Cluster *discardSet = (Cluster *) malloc(numInitialCluster * sizeof(Cluster));
    if (!discardSet) {
        printf("Allocazione memoria per i discard set fallita\n");
        exit(MEM_ALLOC_ERROR);
    }
    for (int i = 0; i < numInitialCluster; ++i) {
        discardSet[i].id = i;
        discardSet[i].size = 0;
        for (int j = 0; j < DIMENSION; ++j) {
            discardSet[i].variance[j] = 0;

        }
    }

    /* Calcolo della memoria a disposizione */

    int avaiableMemory = computeAvaiableMemory();/* numero di punti che possono essere ospitati in memoria */
    int chunkSize = avaiableMemory;
    if (avaiableMemory == 0) {
        printf("Memoria insufficiente");
        free(discardSet), discardSet = NULL;
        exit(NOT_ENOUGHT_MEMORY);
    }

    Point *points = (Point *) malloc(chunkSize * sizeof(Point));
    if (!points) {
        free(discardSet), discardSet = NULL;
        printf("Allocazione memoria per i points set fallita\n");
        exit(MEM_ALLOC_ERROR);
    }

    FILE *dataset = fopen(INPUT_FILE, READ_ONLY);
    if (!dataset) {
        free(discardSet), discardSet = NULL;
        free(points), points = NULL;
        printf("Impossibile aprire il file\n");
        exit(FILE_OPENING_ERROR);
    }

    int numPoint = readPoint(dataset, points, chunkSize, 0);
    if (numPoint < 0) {
        free(discardSet), discardSet = NULL;
        free(points), points = NULL;
        fclose(dataset), dataset = NULL;
        printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
        exit(numPoint);
    }

    /* Eelezione dei centroidi */

    InitResult iRes = init(discardSet, numInitialCluster, points, numPoint);
    if (iRes.err < 0) {
        points = NULL;
        discardSet = NULL;
        fclose(dataset), dataset = NULL;
        exit(iRes.err);
    }
    points = iRes.points;
    numPoint = iRes.numPoints;
    iRes.points = NULL;

    avaiableMemory = computeAvaiableMemory() - numPoint; /*  - numPoint simula la memoria che si riempe */
    if (avaiableMemory == 0) {
        printf("Memoria insufficiente");
        free(discardSet), discardSet = NULL;
        free(points), points = NULL;
        fclose(dataset), dataset = NULL;
        exit(NOT_ENOUGHT_MEMORY);
    }
    chunkSize = avaiableMemory + numPoint;

    Point *temp = (Point *) realloc(points, chunkSize * sizeof(Point));
    if (!temp) {
        free(discardSet), discardSet = NULL;
        free(points), points = NULL;
        fclose(dataset), dataset = NULL;
        printf("allocazione memoria fallita\n");
        exit(MEM_ALLOC_ERROR);
    }
    points = temp, temp = NULL;

    numPoint = readPoint(dataset, points, avaiableMemory, numPoint);
    if (numPoint < 0) {
        free(discardSet), discardSet = NULL;
        free(points), points = NULL;
        fclose(dataset), dataset = NULL;
        printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
        exit(numPoint);
    } else if (numPoint < chunkSize) {
        temp = (Point *) realloc(points, numPoint * sizeof(Point));
        if (!temp) {
            free(discardSet), discardSet = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            printf("allocazione memoria fallita\n");
            exit(MEM_ALLOC_ERROR);
        }
        points = temp, temp = NULL;
        chunkSize = numPoint;
    }

    /* clustering */

    while (numPoint > 0) {

        int res = clusteringAlongInitialCluster(discardSet, numInitialCluster, points, numPoint);
        if (res < 0) {
            points = NULL;
            discardSet = NULL;
            fclose(dataset), dataset = NULL;
            exit(res);
        }
        numPoint = 0;
        points = NULL;

        /* caircamento chunk successivo */

        avaiableMemory = computeAvaiableMemory() - numPoint;

        /*
         * - numPoint simula la memoria che si riempe anche se in realta, apparte che per la prima iterazione,
         * qui numPoints sarà sempre 0 perchè l'algoritmo elabora per intero ogni chunk, scrivendo su file sia
         * i punti clusterizzati che i punti non clusterizzati
         */

        if (avaiableMemory == 0) {
            printf("Memoria insufficiente");
            free(discardSet), discardSet = NULL;
            fclose(dataset), dataset = NULL;
            exit(NOT_ENOUGHT_MEMORY);
        }
        chunkSize = numPoint + avaiableMemory;

        points = (Point *) malloc(chunkSize * sizeof(Point));
        if (!points) {
            printf("allocazione memoria fallita\n");
            free(discardSet), discardSet = NULL;
            fclose(dataset), dataset = NULL;
            exit(numPoint);
        }

        numPoint = readPoint(dataset, points, avaiableMemory, numPoint);
        if (numPoint < 0) {
            printf("lettura da file fallita oppure non ci sono punti nel dataset\n");
            free(discardSet), discardSet = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            exit(numPoint);
        } else if (numPoint < chunkSize && numPoint > 0) {
            temp = (Point *) realloc(points, numPoint * sizeof(Point));
            if (!temp) {
                free(discardSet), discardSet = NULL;
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                printf("allocazione memoria fallita\n");
                exit(numPoint);
            }
            points = temp, temp = NULL;
            chunkSize = numPoint;
        }
    }

    free(discardSet), discardSet = NULL;
    fclose(dataset), dataset = NULL;

    /*
     * arrivato qui, points è allocato con dimensione chunkSize ma non contiene punti, ovvero numPoint = 0
     * quindi deve leggere avaiableMemory punti nella prossima lettura
     * clustering sugli outlier
     * */

    dataset = fopen(OUTLIER_FILE, READ_ONLY);
    if (!dataset) {
        printf("Errore nell'apertura del file %s\n", OUTLIER_FILE);
        free(points), points = NULL;
        exit(FILE_OPENING_ERROR);
    }

    int numCluster = 0;
    Cluster *compressedSet = NULL;
    numPoint = readPoint(dataset, points, avaiableMemory, 0);
    if (numPoint < 0) {
        printf("Impossibile leggere da file %s\n", OUTLIER_FILE);
        free(points), points = NULL;
        fclose(dataset), dataset = NULL;
        exit(numPoint);
    } else if (numPoint < avaiableMemory) {
        temp = (Point *) realloc(points, numPoint * sizeof(Point));
        if (!temp) {
            printf("allocazione memoria fallita\n");
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            exit(numPoint);
        }
        points = temp, temp = NULL;
    }
    chunkSize = numPoint;

    while (numPoint > 0) {

        /* clustering */

        ClusteringResult cluRes = clustringAlongNewCluster(compressedSet, numCluster, points, numPoint);
        if (cluRes.err < 0) {
            compressedSet = NULL;
            points = NULL;
            fclose(dataset), dataset = NULL;
            exit(cluRes.err);
        }
        points = cluRes.outlier;
        numPoint = cluRes.numOutlier;
        cluRes.outlier = NULL;

        /* generazione nuovi cluster */

        GNCResult gncRes = generateNewCompressed(points, numPoint, numInitialCluster + numCluster);
        if (gncRes.err < 0) {
            points = NULL;
            free(compressedSet), compressedSet = NULL;
            fclose(dataset), dataset = NULL;
            exit(gncRes.err);
        }

        Cluster *newCluster = gncRes.cluster;
        gncRes.cluster = NULL;
        points = gncRes.outlier;
        gncRes.outlier = NULL;
        int numNewCluster = gncRes.numCluster;
        numPoint = gncRes.numOutlier;

        if (numNewCluster > 0) {

            /* i nuovi cluster vengono aggiunti a quelli già esistenti */

            Cluster *temp_c = (Cluster *) realloc(compressedSet, (numCluster + numNewCluster) * sizeof(Cluster));
            if (!temp_c) {
                printf("Impossibile allocare memoria\n");
                free(compressedSet), compressedSet = NULL;
                free(newCluster), newCluster = NULL;
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                exit(MEM_ALLOC_ERROR);
            }
            compressedSet = temp_c, temp_c = NULL;

            memcpy(compressedSet + numCluster, newCluster, numNewCluster * sizeof(Cluster));
            numCluster += numNewCluster;
            free(newCluster), newCluster = NULL;

        }

        /* caricamento chunk successivo */

        int oNumPoint = numPoint;
        avaiableMemory = computeAvaiableMemory() - numPoint; /* - numPoint simula la memoria che si occupa */
        chunkSize = numPoint + avaiableMemory;

        if (avaiableMemory == 0) {
            printf("Memoria insufficiente\n");
            free(compressedSet), compressedSet = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            exit(NOT_ENOUGHT_MEMORY);
        }
        temp = (Point *) realloc(points, chunkSize * sizeof(Point));
        if (!temp) {
            printf("allocazione memoria fallita\n");
            free(compressedSet), compressedSet = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            exit(MEM_ALLOC_ERROR);
        }
        points = temp, temp = NULL;
        numPoint = readPoint(dataset, points, avaiableMemory, numPoint);
        if (numPoint < 0) {
            printf("Impossibile leggere da file %s\n", OUTLIER_FILE);
            free(compressedSet), compressedSet = NULL;
            free(points), points = NULL;
            fclose(dataset), dataset = NULL;
            exit(FILE_OPENING_ERROR);
        } else if (numPoint == oNumPoint) break; /* non ci sono nuovi punti da elaborare */
        else if (numPoint < chunkSize) {
            temp = (Point *) realloc(points, numPoint * sizeof(Point));
            if (!temp) {
                printf("allocazione memoria fallita\n");
                free(compressedSet), compressedSet = NULL;
                free(points), points = NULL;
                fclose(dataset), dataset = NULL;
                exit(MEM_ALLOC_ERROR);
            }
            points = temp, temp = NULL;
            chunkSize = numPoint;
        }

    }

    free(compressedSet), compressedSet = NULL;
    fclose(dataset), dataset = NULL;

    /* vengono scritti gli outlier */

    FILE *outputfile = fopen(OUTPUT_FILE, APPEND_MODE);
    if (!outputfile) {
        free(points), points = NULL;
        exit(FILE_OPENING_ERROR);
    }

    for (int i = 0; i < numPoint; ++i) fprintf(outputfile, "%d,OUTLIER\n", points[i].id);

    fclose(outputfile), outputfile = NULL;
    free(points), points = NULL;
    remove(OUTLIER_FILE);

    return 0;
}
int computeAvaiableMemory() {
    return 35000;
}

int readPoint(FILE *fp, Point *points, int to_read, int o_size) {
    int read = 0;

    //while (read < (to_read - o_size) && (*np_file) > 0) {
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

    for (int clusterIdx = 0; clusterIdx < numCluster; ++clusterIdx) {
        int pointIndex = -1; /* indice del punto da assegnare all'i-esimo cluster*/

        /*todo srand è commentato per fare in modo che ogni run sia uguale*/


        if (!clusterIdx) srand(time(NULL)), pointIndex = rand() % numPoint; //generazione numero casuale tra 0 e num_points - 1 per il primo cluster

        else { /* cluster successivi */
            for (int j = 0; j < numPoint; ++j) {
                int check = 1;

                /* se il j-esimo punto è sufficientemente lontano da dai centri dei cluster già inizializzati
                 * allora viene scelto come centro per inizializzare il cluster di indice cluser_idx*/

                for (int k = 0; k < clusterIdx; ++k) {
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

        discardSet[clusterIdx].size++;
        for (int j = 0; j < DIMENSION; j++) {
            discardSet[clusterIdx].coord[j] = points[pointIndex].coords[j];
            discardSet[clusterIdx].squared_coord[j] = pow(points[pointIndex].coords[j], 2);
        }

        /* Scrittura su file del punto */

        fprintf(outputFile, "%d,%d\n", points[pointIndex].id,
                discardSet[clusterIdx].id); /* scrittura del punto in memoria secondaria */

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

int clusteringAlongInitialCluster(Cluster *discardSet, int numCluster, Point *points, int numPoint) {

    FILE *outputFile = fopen(OUTPUT_FILE, APPEND_MODE);
    if (!outputFile) {
        printf("Non è stato possibile creare o aprire il file %s\n", OUTPUT_FILE);
        free(discardSet), discardSet = NULL;
        free(points), points = NULL;
        return MEM_ALLOC_ERROR;
    }

    FILE *outlierFile = fopen(OUTLIER_FILE, APPEND_MODE);
    if (!outlierFile) {
        printf("Non è stato possibile creare o aprire il file %s\n", OUTLIER_FILE);
        free(discardSet), discardSet = NULL;
        free(points), points = NULL;
        fclose(outputFile), outputFile = NULL;
        return MEM_ALLOC_ERROR;
    }

    for (int pointIndex = 0; pointIndex < numPoint; ++pointIndex) { /* index: indice del punto */
        double min_distance = INFINITY;
        int cluster_idx = -1; /* indice del cluster scelto */

        /* valutazione dell'appartenenza dell'i-esimo punto all j-esimo discard set */

        for (int j = 0; j < numCluster; ++j) {

            /* disatnza del punto i dal cluster j */

            double distance = euclideanDistance(points[pointIndex], discardSet[j].coord);

            /* se viene trovato un candidato mogliore */

            if (distance < min_distance && distance < threshold) {
                min_distance = distance;
                cluster_idx = j;
            }
        }

        if (cluster_idx != -1) { /* punto assegnato a un cluster */

            /* aggiornamento statistiche cluster */

            int n = ++discardSet[cluster_idx].size; /* numero di punti nel cluster prima di assegnare il nuovo punto */

            /* aggiornamento delle statistiche del cluster */

            for (int j = 0; j < DIMENSION; ++j) {

                /* sum[i]/n */
                discardSet[cluster_idx].coord[j] =
                        ((discardSet[cluster_idx].coord[j] * n) + points[pointIndex].coords[j]) / (n + 1);

                /* sumq[i]/n */
                discardSet[cluster_idx].squared_coord[j] =
                        ((discardSet[cluster_idx].squared_coord[j] * n) + pow(points[pointIndex].coords[j], 2)) /
                        (n + 1);

                /* varianza: la varianza è calcolata come sumq[i]/n - (sum[i]/n)^2 */
                discardSet[cluster_idx].variance[j] =
                        discardSet[cluster_idx].squared_coord[j] - pow(discardSet[cluster_idx].coord[j], 2);
            }

            /* il punto viene scritto su file e rimosso dalla memoria principale */

            fprintf(outputFile, "%d,%d\n", points[pointIndex].id, cluster_idx);
        } else {
            /* il punto viene scritto su file */

            fprintf(outlierFile, "%d", points[pointIndex].id);
            for (int i = 0; i < DIMENSION; ++i) fprintf(outlierFile, ",%f", points[pointIndex].coords[i]);
            fprintf(outlierFile, "\n");

        }

        /* Il punto viene rimosso dalla memoria */

        memmove(points + pointIndex, points + pointIndex + 1, (numPoint - pointIndex - 1) * sizeof(Point));
        numPoint--;

        Point *temp = (Point *) realloc(points, numPoint * sizeof(Point));
        if (!temp) {
            printf("Allocazione della memoria fallita\n");
            free(points), points = NULL;
            free(discardSet), discardSet = NULL;
            fclose(outlierFile), outlierFile = NULL;
            fclose(outputFile), outputFile = NULL;
            return MEM_ALLOC_ERROR;
        }
        points = temp, temp = NULL;

        pointIndex--; /* altrimenti si saltano punti */
    }

    /* alla fine di questa fase point sarà vuoto */

    free(points);

    fclose(outlierFile), outlierFile = NULL;
    fclose(outputFile), outlierFile = NULL;
    return 0;

}

ClusteringResult clustringAlongNewCluster(Cluster *clusters, int numCluster, Point *points, int numPoints) {

    ClusteringResult res;
    res.outlier = NULL;
    res.numOutlier = 0;
    res.err = 0;

    if (numCluster == 0) {
        res.outlier = points;
        res.numOutlier = numPoints;
        return res;
    }

    Point *outlier = NULL;
    int numOutlier = 0;

    FILE *outputFile = fopen(OUTPUT_FILE, APPEND_MODE);
    if (!outputFile) {
        printf("Impossibile aprire il file %s", OUTPUT_FILE);
        free(clusters), clusters = NULL;
        free(points), points = NULL;
        res.err = FILE_OPENING_ERROR;
        return res;
    }

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

            /* Il punto viene scritto su file */

            fprintf(outputFile, "%d,%d\n", points[pointIndex].id, clusters[clusterIdx].id);
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
                fclose(outputFile), outputFile = NULL;
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
            if (outlier)
                free(outlier), outlier = NULL;  /* qunado arriva qui potrebbe non aver ancora inizializzato outlier */
            fclose(outputFile), outputFile = NULL;
            res.err = MEM_ALLOC_ERROR;
            return res;
        }
        points = temp, temp = NULL;

        pointIndex--; /* altrimenti si saltano punti */
    }

    fclose(outputFile), outputFile = NULL;
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
    res.numCluster = 0;
    res.outlier = 0;
    res.err = 0;

    bool convergence = false;
    if (numPoints == 0) return res;

    int k;
    double wcss_prev = 0, wcss = INFINITY;

    /* ricerca del range [k/2, k] in cui si trova il valore ottimo di k, la ricerca si ferma quando wcss converge */

    for (k = 1; k < numPoints; k *= 2) {
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

    FILE *outputFile = fopen(OUTPUT_FILE, APPEND_MODE);
    if (!outputFile) {
        printf("Impossibile aprire il file %s\n", OUTPUT_FILE);
        free(points), points = NULL;
        res.err = FILE_OPENING_ERROR;
        return res;
    }
    NewClusterResult ncRes = kmeans(points, numPoints, k, startingId, true);
    if (ncRes.err < 0) {
        points = NULL;
        fclose(outputFile), outputFile = NULL;
        res.err = ncRes.err;
        return res;
    }

    points = NULL;
    CompressedSet *clusters = ncRes.newClusters;
    ncRes.newClusters = NULL;
    Point *outlier = ncRes.outlier;
    ncRes.outlier = NULL;
    int numOutlier = ncRes.numOutlier;

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
             * Cluster non valido: il cluster viene rimosso.
             * I punti appartenenti al cluster vengono salvati in outlier
             */

            Point *temp = (Point *) realloc(outlier, (numOutlier + clusters[clusterIndex].size) * sizeof(Point));
            if (!temp) {
                printf("Impossibile allocare la memoria\n");
                for (int j = 0; j < k; ++j) free(clusters[j].points), clusters[j].points = NULL;
                free(clusters), clusters = NULL;
                if (outlier) free(outlier), outlier = NULL; /* arrivato qui outlier potrebbe essere vuoto */
                fclose(outputFile), outputFile = NULL;
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
                fclose(outputFile), outputFile = NULL;
                res.err = MEM_ALLOC_ERROR;
                return res;
            }
            clusters = temp_cs, temp_cs = NULL;

            clusterIndex--; /* altrimenti vengono saltati alcuni cluster */
        } else {

            /* cambio l'id del cluster */
            clusters[clusterIndex].id = startingId;
            startingId++;
            /* il cluster è valido: i punti vengono scritti su file */

            for (int j = 0; j < clusters[clusterIndex].size; ++j)
                fprintf(outputFile, "%d,%d\n", clusters[clusterIndex].points[j].id, clusters[clusterIndex].id);

            free(clusters[clusterIndex].points), clusters[clusterIndex].points = NULL;

        }
    }

    fclose(outputFile), outputFile = NULL;

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
    res.outlier = outlier;
    outlier = NULL;
    res.numCluster = k;
    res.numOutlier = numOutlier;

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

        if (!clusterIdx) srand(time(NULL)), pointIdx = rand() % numPoint;
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
