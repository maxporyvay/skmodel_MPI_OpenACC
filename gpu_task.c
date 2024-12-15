#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <openacc.h>

int np;
int rank;
int process_row;
int process_col;
int total_process_rows = 1;
int total_process_cols = 1;
int upper_process;
int lower_process;
int lefter_process;
int righter_process;
int row_size;
int col_size;
int base_row_size;
int base_col_size;
int addit_rows_cnt;
int addit_cols_cnt;

int i_min;
int i_max;
int j_min;
int j_max;

MPI_Datatype table_col;

int get_process_row_from_rank(int process_rank) {
    return process_rank / total_process_cols;
}

int get_process_col_from_rank_and_row(int process_rank, int row) {
    return process_rank - row * total_process_cols;
}

int get_process_rank_from_row_and_col(int row, int col) {
    return row * total_process_cols + col;
}

int get_upper_process(int row, int col) {
    if (row == 0) {
        return -1;
    }
    return get_process_rank_from_row_and_col(row - 1, col);
}

int get_lower_process(int row, int col) {
    if (row == total_process_rows - 1) {
        return -1;
    }
    return get_process_rank_from_row_and_col(row + 1, col);
}

int get_lefter_process(int row, int col) {
    if (col == 0) {
        return -1;
    }
    return get_process_rank_from_row_and_col(row, col - 1);
}

int get_righter_process(int row, int col) {
    if (col == total_process_cols - 1) {
        return -1;
    }
    return get_process_rank_from_row_and_col(row, col + 1);
}

int get_process_row_size(int process_row) {
    if (process_row + 1 <= addit_rows_cnt) {
        return base_row_size + 1;
    }
    return base_row_size;
}

int get_process_col_size(int process_col) {
    if (process_col + 1 <= addit_cols_cnt) {
        return base_col_size + 1;
    }
    return base_col_size;
}

// start variant-specific constants

const int A1 = -2;
const int B1 = 2;
const int A2 = -2;
const int B2 = 1;

// finish variant-specific constants

int M;
int N;
double delta;
double err_rate;

double *restrict a = NULL;
double *restrict b = NULL;
double *restrict F = NULL;
double *restrict r = NULL;
double *restrict Ar = NULL;
double *restrict w = NULL;

MPI_Request* requests = NULL;
MPI_Status* statuses = NULL;

double h1;
double h2;
double h_12;
double eps;
double div_1_eps;

int iter_cnt = 0;

// start variant-specific functions

double calc_a_elem(double x_i_minus_half, double y_j_minus_half, double y_j_plus_half) {
    double y_higher_border;
    double y_lower_border;
    if (x_i_minus_half > -2.0 && x_i_minus_half < -1.0) {
        y_lower_border = -2.0 - x_i_minus_half;
        y_higher_border = 2.0 + x_i_minus_half;
    } else if (x_i_minus_half >= -1.0 && x_i_minus_half < 0.0) {
        y_lower_border = -2.0 - x_i_minus_half;
        y_higher_border = 1.0;
    } else if (x_i_minus_half >= 0.0 && x_i_minus_half < 1.0) {
        y_lower_border = -2.0 + x_i_minus_half;
        y_higher_border = 1.0;
    } else if (x_i_minus_half >= 1.0 && x_i_minus_half < 2.0) {
        y_lower_border = -2.0 + x_i_minus_half;
        y_higher_border = 2.0 - x_i_minus_half;
    } else {
        return div_1_eps;
    }

    double l;
    if (y_j_minus_half >= y_higher_border || y_j_plus_half <= y_lower_border) {
        return div_1_eps;
    } else if (y_j_minus_half <= y_higher_border && y_j_plus_half >= y_higher_border && y_j_minus_half >= y_lower_border) {
        l = y_higher_border - y_j_minus_half;
    } else if (y_j_plus_half >= y_higher_border && y_j_minus_half <= y_lower_border) {
        l = y_higher_border - y_lower_border;
    } else if (y_j_minus_half <= y_lower_border && y_j_plus_half <= y_higher_border && y_j_plus_half >= y_lower_border) {
        l = y_j_plus_half - y_lower_border;
    } else if (y_j_plus_half <= y_higher_border && y_j_minus_half >= y_lower_border) {
        return 1.0;
    }

    return l / h2 + (1 - l / h2) * div_1_eps;
}

double calc_b_elem(double x_i_minus_half, double x_i_plus_half, double y_j_minus_half) {
    double x_righter_border;
    double x_lefter_border;
    if (y_j_minus_half > -2.0 && y_j_minus_half < 0.0) {
        x_lefter_border = -2.0 - y_j_minus_half;
        x_righter_border = 2.0 + y_j_minus_half;
    } else if (y_j_minus_half >= 0.0 && y_j_minus_half < 1.0) {
        x_lefter_border = -2.0 + y_j_minus_half;
        x_righter_border = 2.0 - y_j_minus_half;
    } else {
        return div_1_eps;
    }

    double l;
    if (x_i_minus_half >= x_righter_border || x_i_plus_half <= x_lefter_border) {
        return div_1_eps;
    } else if (x_i_minus_half <= x_righter_border && x_i_plus_half >= x_righter_border && x_i_minus_half >= x_lefter_border) {
        l = x_righter_border - x_i_minus_half;
    } else if (x_i_plus_half >= x_righter_border && x_i_minus_half <= x_lefter_border) {
        l = x_righter_border - x_lefter_border;
    } else if (x_i_minus_half <= x_lefter_border && x_i_plus_half <= x_righter_border && x_i_plus_half >= x_lefter_border) {
        l = x_i_plus_half - x_lefter_border;
    } else if (x_i_plus_half <= x_righter_border && x_i_minus_half >= x_lefter_border) {
        return 1.0;
    }

    return l / h1 + (1 - l / h1) * div_1_eps;
}

double calc_F_elem(double x_i_minus_half, double x_i_plus_half, double y_j_minus_half, double y_j_plus_half) {
    double y_higher_lefter_border;
    double y_lower_lefter_border;
    if (x_i_minus_half > -2.0 && x_i_minus_half < -1.0) {
        y_lower_lefter_border = -2.0 - x_i_minus_half;
        y_higher_lefter_border = 2.0 + x_i_minus_half;
    } else if (x_i_minus_half >= -1.0 && x_i_minus_half < 0.0) {
        y_lower_lefter_border = -2.0 - x_i_minus_half;
        y_higher_lefter_border = 1.0;
    } else if (x_i_minus_half >= 0.0 && x_i_minus_half < 1.0) {
        y_lower_lefter_border = -2.0 + x_i_minus_half;
        y_higher_lefter_border = 1.0;
    } else if (x_i_minus_half >= 1.0 && x_i_minus_half < 2.0) {
        y_lower_lefter_border = -2.0 + x_i_minus_half;
        y_higher_lefter_border = 2.0 - x_i_minus_half;
    } else if (x_i_minus_half <= -2.0) {
        y_lower_lefter_border = 0.0;
        y_higher_lefter_border = 0.0;
    } else {
        return 0.0;
    }

    double y_higher_righter_border;
    double y_lower_righter_border;
    if (x_i_plus_half > -2.0 && x_i_plus_half < -1.0) {
        y_lower_righter_border = -2.0 - x_i_plus_half;
        y_higher_righter_border = 2.0 + x_i_plus_half;
    } else if (x_i_plus_half >= -1.0 && x_i_plus_half < 0.0) {
        y_lower_righter_border = -2.0 - x_i_plus_half;
        y_higher_righter_border = 1.0;
    } else if (x_i_plus_half >= 0.0 && x_i_plus_half < 1.0) {
        y_lower_righter_border = -2.0 + x_i_plus_half;
        y_higher_righter_border = 1.0;
    } else if (x_i_plus_half >= 1.0 && x_i_plus_half < 2.0) {
        y_lower_righter_border = -2.0 + x_i_plus_half;
        y_higher_righter_border = 2.0 - x_i_plus_half;
    } else if (x_i_plus_half >= 2.0) {
        y_lower_righter_border = 0.0;
        y_higher_righter_border = 0.0;
    } else {
        return 0.0;
    }

    double x_lower_righter_border;
    double x_lower_lefter_border;
    if (y_j_minus_half > -2.0 && y_j_minus_half < 0.0) {
        x_lower_lefter_border = -2.0 - y_j_minus_half;
        x_lower_righter_border = 2.0 + y_j_minus_half;
    } else if (y_j_minus_half >= 0.0 && y_j_minus_half < 1.0) {
        x_lower_lefter_border = -2.0 + y_j_minus_half;
        x_lower_righter_border = 2.0 - y_j_minus_half;
    } else if (y_j_minus_half <= -2.0) {
        x_lower_lefter_border = 0.0;
        x_lower_righter_border = 0.0;
    } else {
        return 0.0;
    }

    double x_higher_righter_border;
    double x_higher_lefter_border;
    if (y_j_plus_half > -2.0 && y_j_plus_half < 0.0) {
        x_higher_lefter_border = -2.0 - y_j_plus_half;
        x_higher_righter_border = 2.0 + y_j_plus_half;
    } else if (y_j_plus_half >= 0.0 && y_j_plus_half < 1.0) {
        x_higher_lefter_border = -2.0 + y_j_plus_half;
        x_higher_righter_border = 2.0 - y_j_plus_half;
    } else if (y_j_plus_half >= 1.0) {
        x_higher_lefter_border = -1.0;
        x_higher_righter_border = 1.0;
    } else {
        return 0.0;
    }

    double S;
    if (y_j_plus_half <= y_higher_righter_border && y_j_minus_half >= y_lower_righter_border &&
        y_j_plus_half <= y_higher_lefter_border && y_j_minus_half >= y_lower_lefter_border &&
        x_i_plus_half <= x_higher_righter_border && x_i_minus_half >= x_higher_lefter_border &&
        x_i_plus_half <= x_lower_righter_border && x_i_minus_half >= x_lower_lefter_border) {  // whole П is inside D
        return 1.0;
    } else if (x_i_minus_half < x_higher_lefter_border && y_j_plus_half > y_higher_lefter_border) {  // Q II
        if (x_i_plus_half <= x_higher_lefter_border) {  // higher side of П is outside D
            if (y_j_minus_half >= y_higher_lefter_border) {  // lefter side of П is outside D
                if (x_i_plus_half > x_lower_lefter_border) {  // triangle
                    S = 0.5 * (x_i_plus_half - x_lower_lefter_border) * (y_higher_righter_border - y_j_minus_half);
                } else {  // whole П is outside D
                    return 0.0;
                }
            } else {  // lefter side of П isn't outside D => trapeze
                S = h1 * (y_higher_lefter_border - 2 * y_j_minus_half + y_higher_righter_border) * 0.5;
            }
        } else if (y_j_minus_half >= y_higher_lefter_border) {  // lefter side of П is outside D while higher side isn't => trapeze
            S = h2 * (x_higher_lefter_border - 2 * x_i_plus_half + x_lower_lefter_border) * 0.5;
        } else {  // both higher and lefter sides of П are partly inside D => mes(П) - outer_triangle
            S = h_12 - 0.5 * (y_j_plus_half - y_higher_lefter_border) * (x_higher_lefter_border - x_i_minus_half);
        }
    } else if (x_i_minus_half < x_lower_lefter_border && y_j_minus_half < y_lower_lefter_border) {  // Q III
        if (x_i_plus_half <= x_lower_lefter_border) {  // lower side of П is outside D
            if (y_j_plus_half <= y_lower_lefter_border) {  // lefter side of П is outside D
                if (x_i_plus_half > x_higher_lefter_border) {  // triangle
                    S = 0.5 * (x_i_plus_half - x_higher_lefter_border) * (y_j_plus_half - y_lower_righter_border);
                } else {  // whole П is outside D
                    return 0.0;
                }
            } else {  // lefter side of П isn't outside D => trapeze
                S = h1 * (2 * y_j_plus_half - y_lower_righter_border - y_lower_lefter_border) * 0.5;
            }
        } else if (y_j_plus_half <= y_lower_lefter_border) {  // lefter side of П is outside D while lower side isn't => trapeze
            S = h2 * (x_higher_lefter_border - 2 * x_i_plus_half + x_lower_lefter_border) * 0.5;
        } else {  // both lower and lefter sides of П are partly inside D => mes(П) - outer_triangle
            S = h_12 - 0.5 * (y_lower_lefter_border - y_j_minus_half) * (x_lower_lefter_border - x_i_minus_half);
        }
    } else if (x_i_plus_half > x_higher_righter_border && y_j_plus_half > y_higher_righter_border) {  // Q I
        if (x_i_minus_half >= x_higher_righter_border) {  // higher side of П is outside D
            if (y_j_minus_half >= y_higher_righter_border) {  // righter side of П is outside D
                if (x_i_minus_half < x_lower_righter_border) {  // triangle
                    S = 0.5 * (x_lower_righter_border - x_i_minus_half) * (y_higher_lefter_border - y_j_minus_half);
                } else {  // whole П is outside D
                    return 0.0;
                }
            } else {  // righter side of П isn't outside D => trapeze
                S = h1 * (y_higher_lefter_border - 2 * y_j_minus_half + y_higher_righter_border) * 0.5;
            }
        } else if (y_j_minus_half >= y_higher_righter_border) {  // righter side of П is outside D while higher side isn't => trapeze
            S = h2 * (x_higher_righter_border - 2 * x_i_minus_half + x_lower_righter_border) * 0.5;
        } else {  // both higher and righter sides of П are partly inside D => mes(П) - outer_triangle
            S = h_12 - 0.5 * (y_j_plus_half - y_higher_righter_border) * (x_i_plus_half - x_higher_righter_border);
        }
    } else if (x_i_plus_half > x_lower_righter_border && y_j_minus_half < y_lower_righter_border) {  // Q IV
        if (x_i_minus_half >= x_lower_righter_border) {  // lower side of П is outside D
            if (y_j_plus_half <= y_lower_righter_border) {  // righter side of П is outside D
                if (x_i_minus_half < x_higher_righter_border) {  // triangle
                    S = 0.5 * (x_higher_righter_border - x_i_minus_half) * (y_j_plus_half - y_lower_lefter_border);
                } else {  // whole П is outside D
                    return 0.0;
                }
            } else {  // righter side of П isn't outside D => trapeze
                S = h1 * (2 * y_j_plus_half - y_lower_lefter_border - y_lower_righter_border) * 0.5;
            }
        } else if (y_j_plus_half <= y_lower_righter_border) {  // righter side of П is outside D while lower side isn't => trapeze
            S = h2 * (x_higher_righter_border - 2 * x_i_plus_half + x_lower_righter_border) * 0.5;
        } else {  // both lower and righter sides of П are partly inside D => mes(П) - outer_triangle
            S = h_12 - 0.5 * (y_lower_righter_border - y_j_minus_half) * (x_i_plus_half - x_lower_righter_border);
        }
    }

    return S / h_12;
}

// finish variant-specific functions

void neighbours_send_recv(double* mat) {
    int req_count = 0;
    if (process_row > 0) {
        req_count += 2;
        MPI_Isend(mat + col_size + 2 + 1, col_size, MPI_DOUBLE, upper_process, rank * 100, MPI_COMM_WORLD, &requests[req_count - 2]);
        MPI_Irecv(mat + 1, col_size, MPI_DOUBLE, upper_process, upper_process * 100 + 1, MPI_COMM_WORLD, &requests[req_count - 1]);
    }
    if (process_row < total_process_rows - 1) {
        req_count += 2;
        MPI_Isend(mat + (col_size + 2) * row_size + 1, col_size, MPI_DOUBLE, lower_process, rank * 100 + 1, MPI_COMM_WORLD, &requests[req_count - 2]);
        MPI_Irecv(mat + (col_size + 2) * (row_size + 1) + 1, col_size, MPI_DOUBLE, lower_process, lower_process * 100, MPI_COMM_WORLD, &requests[req_count - 1]);
    }
    if (process_col > 0) {
        req_count += 2;
        MPI_Isend(mat + col_size + 2 + 1, 1, table_col, lefter_process, rank * 100 + 2, MPI_COMM_WORLD, &requests[req_count - 2]);
        MPI_Irecv(mat + col_size + 2, 1, table_col, lefter_process, lefter_process * 100 + 3, MPI_COMM_WORLD, &requests[req_count - 1]);
    }
    if (process_col < total_process_cols - 1) {
        req_count += 2;
        MPI_Isend(mat + col_size + 2 + col_size, 1, table_col, righter_process, rank * 100 + 3, MPI_COMM_WORLD, &requests[req_count - 2]);
        MPI_Irecv(mat + col_size + 2 + col_size + 1, 1, table_col, righter_process, righter_process * 100 + 2, MPI_COMM_WORLD, &requests[req_count - 1]);
    }
    MPI_Waitall(req_count, requests, statuses);
}

void solve() {
    for (int i = 1; i < row_size + 2; i++) {
		for (int j = 1; j < col_size + 2; j++) {
            int global_j = process_col * base_col_size + j;
            int global_i = process_row * base_row_size + i;
            if (process_row + 1 <= addit_rows_cnt) {
                global_i += process_row;
            } else {
                global_i += addit_rows_cnt;
            }
            if (process_col + 1 <= addit_cols_cnt) {
                global_j += process_col;
            } else {
                global_j += addit_cols_cnt;
            }
            double x_i = A1 + global_j * h1;
            double y_j = B2 - global_i  * h2;  // because y-axis is inverted
            double x_i_minus_half = x_i - 0.5 * h1;
            double x_i_plus_half = x_i + 0.5 * h1;
            double y_j_minus_half = y_j - 0.5 * h2;
            double y_j_plus_half = y_j + 0.5 * h2;
			a[i * (col_size + 2) + j] = calc_a_elem(x_i_minus_half, y_j_minus_half, y_j_plus_half);
            b[i * (col_size + 2) + j] = calc_b_elem(x_i_minus_half, x_i_plus_half, y_j_minus_half);
            F[i * (col_size + 2) + j] = calc_F_elem(x_i_minus_half, x_i_plus_half, y_j_minus_half, y_j_plus_half);
		}
	}

    err_rate = delta;

    MPI_Type_vector(row_size, 1, col_size + 2, MPI_DOUBLE, &table_col);
    MPI_Type_commit(&table_col);

    #pragma acc data copyin(a[0:(row_size + 2) * (col_size + 2)], b[0:(row_size + 2) * (col_size + 2)], F[0:(row_size + 2) * (col_size + 2)], w[0:(row_size + 2) * (col_size + 2)], r[0:(row_size + 2) * (col_size + 2)], Ar[0:(row_size + 2) * (col_size + 2)])
    while (err_rate >= delta) {
        // #pragma acc update host(w[0:(row_size + 2) * (col_size + 2)])
        #pragma acc update host(w[0:(col_size + 2) * 2])
        #pragma acc update host(w[(col_size + 2) * row_size:(col_size + 2) * 2])
        for (int i = 2; i < row_size; i++) {
            #pragma acc update host(w[(col_size + 2) * i + col_size:2])
            #pragma acc update host(w[(col_size + 2) * i:2])
        }
        neighbours_send_recv(w);
        // #pragma acc update device(w[0:(row_size + 2) * (col_size + 2)])
        #pragma acc update device(w[0:(col_size + 2) * 2])
        #pragma acc update device(w[(col_size + 2) * row_size:(col_size + 2) * 2])
        for (int i = 2; i < row_size; i++) {
            #pragma acc update device(w[(col_size + 2) * i + col_size:2])
            #pragma acc update device(w[(col_size + 2) * i:2])
        }
        #pragma acc parallel loop collapse(2)
        for (int i = 1; i < row_size + 1; i++) {
            for (int j = 1; j < col_size + 1; j++) {
                double part_a = (a[(i + 1) * (col_size + 2) + j] * (w[(i + 1) * (col_size + 2) + j] - w[i * (col_size + 2) + j]) - a[i * (col_size + 2) + j] * (w[i * (col_size + 2) + j] - w[(i - 1) * (col_size + 2) + j])) / (-h1 * h1);
                double part_b = (b[i * (col_size + 2) + j + 1] * (w[i * (col_size + 2) + j + 1] - w[i * (col_size + 2) + j]) - b[i * (col_size + 2) + j] * (w[i * (col_size + 2) + j] - w[i * (col_size + 2) + j - 1])) / (-h2 * h2);
                r[i * (col_size + 2) + j] = part_a + part_b - F[i * (col_size + 2) + j];
            }
        }

        // #pragma acc update host(r[0:(row_size + 2) * (col_size + 2)])
        #pragma acc update host(r[0:(col_size + 2) * 2])
        #pragma acc update host(r[(col_size + 2) * row_size:(col_size + 2) * 2])
        for (int i = 2; i < row_size; i++) {
            #pragma acc update host(r[(col_size + 2) * i + col_size:2])
            #pragma acc update host(r[(col_size + 2) * i:2])
        }
        neighbours_send_recv(r);
        // #pragma acc update device(r[0:(row_size + 2) * (col_size + 2)])
        #pragma acc update device(r[0:(col_size + 2) * 2])
        #pragma acc update device(r[(col_size + 2) * row_size:(col_size + 2) * 2])
        for (int i = 2; i < row_size; i++) {
            #pragma acc update device(r[(col_size + 2) * i + col_size:2])
            #pragma acc update device(r[(col_size + 2) * i:2])
        }
        #pragma acc parallel loop collapse(2)
        for (int i = 1; i < row_size + 1; i++) {
            for (int j = 1; j < col_size + 1; j++) {
                double part_a = (a[(i + 1) * (col_size + 2) + j] * (r[(i + 1) * (col_size + 2) + j] - r[i * (col_size + 2) + j]) - a[i * (col_size + 2) + j] * (r[i * (col_size + 2) + j] - r[(i - 1) * (col_size + 2) + j])) / (-h1 * h1);
                double part_b = (b[i * (col_size + 2) + j + 1] * (r[i * (col_size + 2) + j + 1] - r[i * (col_size + 2) + j]) - b[i * (col_size + 2) + j] * (r[i * (col_size + 2) + j] - r[i * (col_size + 2) + j - 1])) / (-h2 * h2);
                Ar[i * (col_size + 2) + j] = part_a + part_b;
            }
        }

        double local_sum_r_r = 0.0;
        #pragma acc parallel loop collapse(2) reduction(+:local_sum_r_r)
        for (int i = 1; i < row_size + 1; i++) {
            for (int j = 1; j < col_size + 1; j++) {
                local_sum_r_r += r[i * (col_size + 2) + j] * r[i * (col_size + 2) + j];
            }
        }
        
        double local_sum_Ar_r = 0.0;
        #pragma acc parallel loop collapse(2) reduction(+:local_sum_Ar_r)
        for (int i = 1; i < row_size + 1; i++) {
            for (int j = 1; j < col_size + 1; j++) {
                local_sum_Ar_r += Ar[i * (col_size + 2) + j] * r[i * (col_size + 2) + j];
            }
        }

        double scalar_mult_r_r;
        MPI_Allreduce(&local_sum_r_r, &scalar_mult_r_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double scalar_mult_Ar_r;
        MPI_Allreduce(&local_sum_Ar_r, &scalar_mult_Ar_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double tau = scalar_mult_r_r / scalar_mult_Ar_r;

        #pragma acc parallel loop collapse(2)
        for (int i = 1; i < row_size + 1; i++) {
            for (int j = 1; j < col_size + 1; j++) {
                w[i * (col_size + 2) + j] -= r[i * (col_size + 2) + j] * tau;
            }
        }
        
        err_rate = sqrt(scalar_mult_r_r) * tau * sqrt(h_12);

        if (rank == 0) {
            // printf("(r, r) = %.20f\n(Ar, r) = %.20f\n", scalar_mult_r_r, scalar_mult_Ar_r);
            // printf("tau = %.20f\n", tau);
            // printf("Error rate = %.20f while delta = %.20f\n", err_rate, delta);
            iter_cnt++;
        }
    }
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        delta = atof(argv[3]);
    }
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int candidate = sqrt(np); candidate > 0; candidate--) {
            if (np % candidate == 0) {
                total_process_rows = candidate;
                total_process_cols = np / candidate;
                break;
            }
        }
    }
    MPI_Bcast(&total_process_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_process_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    process_row = get_process_row_from_rank(rank);
    process_col = get_process_col_from_rank_and_row(rank, process_row);

    base_row_size = (N - 1) / total_process_rows;  // height of row
    base_col_size = (M - 1) / total_process_cols;  // width of col
    addit_rows_cnt = (N - 1) % total_process_rows;
    addit_cols_cnt = (M - 1) % total_process_cols;
    MPI_Bcast(&base_row_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_col_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&addit_rows_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&addit_cols_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    row_size = get_process_row_size(process_row);
    col_size = get_process_col_size(process_col);

    upper_process = get_upper_process(process_row, process_col);
    lower_process = get_lower_process(process_row, process_col);
    lefter_process = get_lefter_process(process_row, process_col);
    righter_process = get_righter_process(process_row, process_col);

    // printf("current process rank: %d\nprocess row: %d\nprocess col: %d\n", rank, process_row, process_col);
    // printf("upper process rank: %d\nlower process rank: %d\nlefter process rank: %d\nrighter process rank: %d\n\n",
    //        upper_process, lower_process, lefter_process, righter_process);
    // printf("width of col: %d\nheight of row: %d\n", col_size, row_size);

    // if (rank == 0) {
    //     printf("total process rows: %d\ntotal process cols: %d\n", total_process_rows, total_process_cols);
    // }

    // printf("current thread rank: %d of %d\n", omp_get_thread_num(), omp_get_num_threads());

    a = malloc((row_size + 2) * (col_size + 2) * sizeof(double));
    b = malloc((row_size + 2) * (col_size + 2) * sizeof(double));
    F = malloc((row_size + 2) * (col_size + 2) * sizeof(double));
    r = malloc((row_size + 2) * (col_size + 2) * sizeof(double));
    Ar = malloc((row_size + 2) * (col_size + 2) * sizeof(double));
    w = malloc((row_size + 2) * (col_size + 2) * sizeof(double));
    for (int i = 0; i < row_size + 2; i++) {
		for (int j = 0; j < col_size + 2; j++) {
			w[i * (col_size + 2) + j] = 0.0;
            r[i * (col_size + 2) + j] = 0.0;
            Ar[i * (col_size + 2) + j] = 0.0;
		}
	}
    requests = malloc(8 * sizeof(MPI_Request));
    statuses = malloc(8 * sizeof(MPI_Status));

    if (rank == 0) {
        h1 = (B1 - A1) / (double)M;
        h2 = (B2 - A2) / (double)N;

        h_12 = h1 * h2;
        double h_max = fmax(h1, h2);
        eps = h_max * h_max;
        div_1_eps = 1.0 / eps;
    }
    MPI_Bcast(&h1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h_12, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&div_1_eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("A1: %d\nB1: %d\n%d <= x <= %d\n", A1, B1, A1, B1);
        printf("A2: %d\nB2: %d\n%d <= y <= %d\n", A2, B2, A2, B2);
        printf("M: %d\nN: %d\n", M, N);
        printf("h1: %.20f\nh2: %.20f\n", h1, h2);
        printf("eps: %.20f\ndelta: %.20f\n\n", eps, delta);
    }

    double time_start;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }
    solve();
    if (rank == 0) {
        double time_elapsed = MPI_Wtime() - time_start;
        printf("\nTime elapsed: %.20f\n", time_elapsed);
        printf("Number of iterations: %d\n", iter_cnt);
    }

    free(a);
    free(b);
    free(F);
    free(r);
    free(Ar);
    free(w);
    free(requests);
    free(statuses);

    MPI_Type_free(&table_col);
    MPI_Finalize();

    return 0;
}
