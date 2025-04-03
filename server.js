const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const path = require('path');
const fs = require('fs');

// Load environment variables
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Create necessary files if they don't exist
const usersFile = 'users.json';
const workoutHistoryFile = 'workout_history.json';

if (!fs.existsSync(usersFile)) {
    fs.writeFileSync(usersFile, JSON.stringify([]));
}

if (!fs.existsSync(workoutHistoryFile)) {
    fs.writeFileSync(workoutHistoryFile, JSON.stringify([]));
}

// Load users and workout history
let users = JSON.parse(fs.readFileSync(usersFile));
let workoutHistory = JSON.parse(fs.readFileSync(workoutHistoryFile));

// Authentication middleware
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }

    jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key', (err, user) => {
        if (err) {
            return res.status(403).json({ error: 'Invalid or expired token' });
        }
        req.user = user;
        next();
    });
}

// Pose correction endpoint
app.post('/api/correct-pose', async (req, res) => {
    try {
        const { pose, exercise } = req.body;
        
        if (!pose || !exercise) {
            return res.status(400).json({ error: 'Pose and exercise are required' });
        }

        // For now, return a simple response
        // In a real implementation, this would use the trained model
        res.json({ 
            correctedPose: pose,
            exerciseConfidence: 0.8
        });
    } catch (error) {
        console.error('Error correcting pose:', error);
        res.status(500).json({ error: 'Failed to correct pose' });
    }
});

// User registration
app.post('/api/register', async (req, res) => {
    try {
        const { username, password } = req.body;

        if (!username || !password) {
            return res.status(400).json({ error: 'Username and password are required' });
        }

        // Check if user already exists
        if (users.find(u => u.username === username)) {
            return res.status(400).json({ error: 'Username already exists' });
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create new user
        const newUser = {
            id: users.length + 1,
            username,
            password: hashedPassword
        };

        users.push(newUser);
        fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));

        res.status(201).json({ message: 'User registered successfully' });
    } catch (error) {
        console.error('Error registering user:', error);
        res.status(500).json({ error: 'Failed to register user' });
    }
});

// User login
app.post('/api/login', async (req, res) => {
    try {
        const { username, password } = req.body;

        if (!username || !password) {
            return res.status(400).json({ error: 'Username and password are required' });
        }

        // Find user
        const user = users.find(u => u.username === username);
        if (!user) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        // Check password
        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        // Generate token
        const token = jwt.sign(
            { id: user.id, username: user.username },
            process.env.JWT_SECRET || 'your-secret-key',
            { expiresIn: '24h' }
        );

        res.json({ token });
    } catch (error) {
        console.error('Error logging in:', error);
        res.status(500).json({ error: 'Failed to log in' });
    }
});

// Get workout history
app.get('/api/workout-history', authenticateToken, (req, res) => {
    try {
        const userHistory = workoutHistory.filter(workout => workout.userId === req.user.id);
        res.json(userHistory);
    } catch (error) {
        console.error('Error getting workout history:', error);
        res.status(500).json({ error: 'Failed to get workout history' });
    }
});

// Save workout
app.post('/api/workout-history', authenticateToken, (req, res) => {
    try {
        const { exercise, score, date } = req.body;

        if (!exercise || !score || !date) {
            return res.status(400).json({ error: 'Exercise, score, and date are required' });
        }

        const newWorkout = {
            id: workoutHistory.length + 1,
            userId: req.user.id,
            exercise,
            score,
            date
        };

        workoutHistory.push(newWorkout);
        fs.writeFileSync(workoutHistoryFile, JSON.stringify(workoutHistory, null, 2));

        res.status(201).json(newWorkout);
    } catch (error) {
        console.error('Error saving workout:', error);
        res.status(500).json({ error: 'Failed to save workout' });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something broke!' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Access the application at http://localhost:${PORT}`);
});