const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');

// Load environment variables
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// MongoDB connection with detailed error logging
console.log('Attempting to connect to MongoDB...');
mongoose.connect('mongodb://127.0.0.1:27017/formsense')
.then(() => {
    console.log('Successfully connected to MongoDB');
})
.catch(err => {
    console.error('MongoDB connection error:', err);
    console.error('Full error details:', JSON.stringify(err, null, 2));
});

// Add connection error handler
mongoose.connection.on('error', err => {
    console.error('MongoDB connection error:', err);
});

// User Schema
const userSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    password: { type: String, required: true }
});

const User = mongoose.model('User', userSchema);

// Workout History Schema
const workoutSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    exercise: { type: String, required: true },
    score: { type: Number, required: true },
    date: { type: Date, required: true }
});

const Workout = mongoose.model('Workout', workoutSchema);

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

// User registration with detailed error logging
app.post('/api/register', async (req, res) => {
    try {
        console.log('Registration attempt with data:', {
            username: req.body.username,
            passwordLength: req.body.password ? req.body.password.length : 0
        });

        const { username, password } = req.body;

        if (!username || !password) {
            console.log('Missing required fields');
            return res.status(400).json({ error: 'Username and password are required' });
        }

        // Check if user already exists
        const existingUser = await User.findOne({ username });
        if (existingUser) {
            console.log('Username already exists:', username);
            return res.status(400).json({ error: 'Username already exists' });
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);
        console.log('Password hashed successfully');

        // Create new user
        const user = new User({
            username,
            password: hashedPassword
        });

        await user.save();
        console.log('User saved successfully:', username);

        res.status(201).json({ message: 'User registered successfully' });
    } catch (error) {
        console.error('Registration error:', error);
        console.error('Full error details:', JSON.stringify(error, null, 2));
        res.status(500).json({ error: 'Failed to register user: ' + error.message });
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
        const user = await User.findOne({ username });
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
            { id: user._id, username: user.username },
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
app.get('/api/workout-history', authenticateToken, async (req, res) => {
    try {
        const workouts = await Workout.find({ userId: req.user.id });
        res.json(workouts);
    } catch (error) {
        console.error('Error getting workout history:', error);
        res.status(500).json({ error: 'Failed to get workout history' });
    }
});

// Save workout
app.post('/api/workout-history', authenticateToken, async (req, res) => {
    try {
        const { exercise, score, date } = req.body;

        if (!exercise || !score || !date) {
            return res.status(400).json({ error: 'Exercise, score, and date are required' });
        }

        const workout = new Workout({
            userId: req.user.id,
            exercise,
            score,
            date: new Date(date)
        });

        await workout.save();
        res.status(201).json(workout);
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

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Access the application at http://localhost:${PORT}`);
});