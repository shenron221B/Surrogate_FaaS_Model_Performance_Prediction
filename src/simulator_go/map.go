package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

// MAP represents a Markovian Arrival Process
type MAP struct {
	D0     Matrix // Transition matrix with no arrivals
	D1     Matrix // Transition matrix with one arrival
	states int    // Number of states
}

// NewMAPFromString reads elements of D0 and D1 concatenated in a list (; is the separator)
func NewMAPFromString(s string) (*MAP, error) {
	tokens := strings.Split(strings.TrimSpace(s), ";")

	if len(tokens)%2 > 0 || len(tokens) < 0 {
		log.Fatal("string should have a even number of elements")
	}

	states := int(math.Sqrt(float64(len(tokens) / 2)))
	if states*states != len(tokens)/2 {
		log.Fatal("D0 and D1 must be square matrices")
	}

	D0 := make([][]float64, states)
	D1 := make([][]float64, states)
	for i := range D0 {
		D0[i] = make([]float64, states)
		D1[i] = make([]float64, states)
	}

	for i, elem := range tokens[:states*states] {
		val, err := strconv.ParseFloat(strings.TrimSpace(elem), 64)
		if err != nil {
			return nil, err
		}
		D0[i/states][i%states] = val
	}
	for i, elem := range tokens[states*states:] {
		val, err := strconv.ParseFloat(strings.TrimSpace(elem), 64)
		if err != nil {
			return nil, err
		}
		D1[i/states][i%states] = val
	}

	return NewMAP(D0, D1)
}

// NewMAP creates a new MAP with given D0 and D1 matrices
func NewMAP(D0, D1 [][]float64) (*MAP, error) {
	if len(D0) != len(D1) || len(D0) == 0 {
		return nil, fmt.Errorf("D0 and D1 must be square matrices of the same size")
	}

	states := len(D0)

	// Validate matrix dimensions
	for i := 0; i < states; i++ {
		if len(D0[i]) != states || len(D1[i]) != states {
			return nil, fmt.Errorf("matrices must be square")
		}
	}

	// Compute generator matrix Q = D0 + D1
	Q := make([][]float64, states)
	for i := 0; i < states; i++ {
		Q[i] = make([]float64, states)
		for j := 0; j < states; j++ {
			Q[i][j] = D0[i][j] + D1[i][j]
		}
	}

	// Validate that Q is a valid generator matrix
	if err := validateGenerator(Q); err != nil {
		return nil, fmt.Errorf("invalid generator matrix: %v", err)
	}

	return &MAP{
		D0:     D0,
		D1:     D1,
		states: states,
	}, nil
}

// validateGenerator checks if Q is a valid generator matrix
func validateGenerator(Q [][]float64) error {
	states := len(Q)
	tolerance := 1e-10

	for i := 0; i < states; i++ {
		rowSum := 0.0
		for j := 0; j < states; j++ {
			if i != j && Q[i][j] < 0 {
				return fmt.Errorf("off-diagonal elements must be non-negative")
			}
			rowSum += Q[i][j]
		}
		if math.Abs(rowSum) > tolerance {
			return fmt.Errorf("row %d sum is %f, should be 0", i, rowSum)
		}
	}
	return nil
}

// Sample generates IATs from the MAP
func (m *MAP) Sample(nSamples int, initialState int, rng *rand.Rand) ([]float64, int, error) {
	if initialState < 0 || initialState >= m.states {
		u := rng.Float64()
		pi := m.StationaryDistribution()
		cumSum := 0.0
		for i := 0; i < m.states; i++ {
			initialState = i
			cumSum += pi[i]
			if u <= cumSum {
				break
			}
		}
	}

	samples := make([]float64, 0, nSamples)
	currentTime := 0.0
	lastArrivalTime := 0.0

	var isArrival bool
	currentState := initialState

	for len(samples) < nSamples {
		// Calculate total rate from current state
		totalRate := -m.D0[currentState][currentState]

		if totalRate <= 0 {
			return samples, currentState, nil
		}

		// Sample time to next event (exponentially distributed)
		nextEventTime := currentTime + rng.ExpFloat64()/totalRate

		currentState, isArrival = m.sampleTransitionAndType(currentState, totalRate, rng)

		if isArrival {
			// Record arrival event
			samples = append(samples, nextEventTime-lastArrivalTime)
			lastArrivalTime = nextEventTime
		}

		currentTime = nextEventTime
	}

	return samples, currentState, nil
}

// sampleTransitionAndType jointly samples the next state and whether it's an arrival
// This is more efficient than separate sampling of transition type and next state
func (m *MAP) sampleTransitionAndType(currentState int, totalRate float64, rng *rand.Rand) (int, bool) {
	if totalRate < 0 {
		panic("invalid total rate")
	}

	u := rng.Float64() * totalRate
	cumulativeRate := 0.0

	// First check D0 transitions (no arrivals)
	for j := 0; j < m.states; j++ {
		if j != currentState {
			cumulativeRate += m.D0[currentState][j]
			if u <= cumulativeRate {
				return j, false // Transition to state j, no arrival
			}
		}
	}

	// Then check D1 transitions (with arrivals)
	for j := 0; j < m.states; j++ {
		cumulativeRate += m.D1[currentState][j]
		if u <= cumulativeRate {
			return j, true // Transition to state j, with arrival
		}
	}

	// Fallback (should not reach here in valid MAP)
	return currentState, false
}

// Matrix represents a 2D matrix
type Matrix [][]float64

// Vector represents a 1D vector
type Vector []float64

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

// NewVector creates a new vector with given size
func NewVector(size int) Vector {
	return make(Vector, size)
}

// add adds two matrices
func (m Matrix) add(other Matrix) Matrix {
	rows, cols := len(m), len(m[0])
	result := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] + other[i][j]
		}
	}
	return result
}

// multiplyVector multiplies matrix by vector
func (m Matrix) multiplyVector(v Vector) Vector {
	rows := len(m)
	result := NewVector(rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < len(v); j++ {
			result[i] += m[i][j] * v[j]
		}
	}
	return result
}

// inverse computes matrix inverse using Gaussian elimination
func (m Matrix) inverse() Matrix {
	n := len(m)
	if n != len(m[0]) {
		panic("Matrix must be square for inversion")
	}

	// Create augmented matrix [A|I]
	augmented := NewMatrix(n, 2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			augmented[i][j] = m[i][j]
		}
		augmented[i][i+n] = 1.0 // Identity matrix
	}

	// Gaussian elimination with partial pivoting
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(augmented[k][i]) > math.Abs(augmented[maxRow][i]) {
				maxRow = k
			}
		}

		// Swap rows
		if maxRow != i {
			augmented[i], augmented[maxRow] = augmented[maxRow], augmented[i]
		}

		// Check for singular matrix
		if math.Abs(augmented[i][i]) < 1e-10 {
			panic("Matrix is singular and cannot be inverted")
		}

		// Scale pivot row
		pivot := augmented[i][i]
		for j := 0; j < 2*n; j++ {
			augmented[i][j] /= pivot
		}

		// Eliminate column
		for k := 0; k < n; k++ {
			if k != i {
				factor := augmented[k][i]
				for j := 0; j < 2*n; j++ {
					augmented[k][j] -= factor * augmented[i][j]
				}
			}
		}
	}

	// Extract inverse matrix
	inverse := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inverse[i][j] = augmented[i][j+n]
		}
	}

	return inverse
}

// dot computes dot product of two vectors
func (v Vector) dot(other Vector) float64 {
	if len(v) != len(other) {
		panic("Vector dimensions don't match for dot product")
	}

	result := 0.0
	for i := range v {
		result += v[i] * other[i]
	}
	return result
}

func (m *MAP) StationaryDistribution() Vector {
	Q := m.D0.add(m.D1)
	return findStationaryDistribution(Q)
}

// findStationaryDistribution finds the stationary distribution of generator matrix Q
func findStationaryDistribution(Q Matrix) Vector {
	n := len(Q)

	// Create system (Q^T)π = 0 with constraint Σπᵢ = 1
	// We replace the last equation with the normalization constraint
	A := NewMatrix(n, n)
	b := NewVector(n)

	// Transpose Q and copy to A (except last row)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n; j++ {
			A[i][j] = Q[j][i] // Transpose
		}
	}

	// Last row is normalization constraint: sum of π = 1
	for j := 0; j < n; j++ {
		A[n-1][j] = 1.0
	}
	b[n-1] = 1.0

	// Solve Aπ = b
	pi := A.inverse().multiplyVector(b)
	return pi
}

// ArrivalRate computes the average arrival rate of a MAP
func (m *MAP) ArrivalRate() float64 {
	pi := m.StationaryDistribution()

	// Step 3: Create vector of ones
	n := m.states
	ones := NewVector(n)
	for i := range ones {
		ones[i] = 1.0
	}

	// Step 4: Compute D1 * ones (row sums of D1)
	d1RowSums := m.D1.multiplyVector(ones)

	// Step 5: Compute π * (D1 * ones) = sum of π[i] * (row sum of D1[i])
	arrivalRate := pi.dot(d1RowSums)

	return arrivalRate
}

func (m *MAP) Rescale(arvRate float64) {
	current := m.ArrivalRate()
	for i := range m.states {
		for j := range m.states {
			m.D0[i][j] *= arvRate / current
			m.D1[i][j] *= arvRate / current
		}
	}
}
