let classifierModel, poseDetector;
let exerciseLabels = [];
let lastPredictions = [];
let selectedExercise = null;
const SMOOTHING_WINDOW_SIZE = 10;
const CONFIDENCE_THRESHOLD = 0.2;
const MIN_POSE_SCORE = 0.25;

const videoElement = document.querySelector(".input_video");
const canvasElement = document.querySelector(".output_canvas");
const predictionBox = document.getElementById("predictionBox");
const formScoreInput = document.getElementById("formScore");
const workoutNameInput = document.getElementById("workoutName");
const exerciseSelectInput = document.getElementById("exerciseSelect");
const exerciseListDatalist = document.getElementById("exerciseList");
const ctx = canvasElement.getContext("2d");

const FORM_RULES = {
  'barbell bench press': {
    checkForm: (keypoints, correctedPose) => {
      const feedback = [];
      let score = 100;
      
      // Get key joints
      const leftShoulder = keypoints[11];
      const rightShoulder = keypoints[12];
      const leftElbow = keypoints[13];
      const rightElbow = keypoints[14];
      const leftWrist = keypoints[15];
      const rightWrist = keypoints[16];
      
      // Get corrected joints
      const correctedLeftShoulder = correctedPose[5];
      const correctedRightShoulder = correctedPose[8];
      const correctedLeftElbow = correctedPose[6];
      const correctedRightElbow = correctedPose[9];
      const correctedLeftWrist = correctedPose[7];
      const correctedRightWrist = correctedPose[10];
      
      // Check shoulder alignment
      const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
      const correctedShoulderDiff = Math.abs(correctedLeftShoulder.y - correctedRightShoulder.y);
      if (shoulderDiff > 0.1) {
        feedback.push('Keep your shoulders level');
        score -= 20;
      }
      
      // Check elbow angle
      const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
      const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
      const correctedLeftElbowAngle = calculateAngle(correctedLeftShoulder, correctedLeftElbow, correctedLeftWrist);
      const correctedRightElbowAngle = calculateAngle(correctedRightShoulder, correctedRightElbow, correctedRightWrist);
      
      if (Math.abs(leftElbowAngle - correctedLeftElbowAngle) > 15 || 
          Math.abs(rightElbowAngle - correctedRightElbowAngle) > 15) {
        feedback.push('Maintain proper elbow angle');
        score -= 20;
      }
      
      // Check wrist position
      const leftWristHeight = leftWrist.y;
      const rightWristHeight = rightWrist.y;
      const correctedLeftWristHeight = correctedLeftWrist.y;
      const correctedRightWristHeight = correctedRightWrist.y;
      
      if (Math.abs(leftWristHeight - correctedLeftWristHeight) > 0.1 ||
          Math.abs(rightWristHeight - correctedRightWristHeight) > 0.1) {
        feedback.push('Keep your wrists straight');
        score -= 20;
      }
      
      return {
        score: Math.max(0, score),
        feedback
      };
    }
  },
  'barbell squat': {
    checkForm: (keypoints, correctedPose) => {
      const feedback = [];
      let score = 100;
      
      // Get key joints
      const leftHip = keypoints[23];
      const rightHip = keypoints[24];
      const leftKnee = keypoints[25];
      const rightKnee = keypoints[26];
      const leftAnkle = keypoints[27];
      const rightAnkle = keypoints[28];
      
      // Get corrected joints
      const correctedLeftHip = correctedPose[11];
      const correctedRightHip = correctedPose[14];
      const correctedLeftKnee = correctedPose[12];
      const correctedRightKnee = correctedPose[15];
      const correctedLeftAnkle = correctedPose[13];
      const correctedRightAnkle = correctedPose[16];
      
      // Check knee alignment
      const leftKneeAlignment = Math.abs(leftKnee.x - leftAnkle.x);
      const rightKneeAlignment = Math.abs(rightKnee.x - rightAnkle.x);
      const correctedLeftKneeAlignment = Math.abs(correctedLeftKnee.x - correctedLeftAnkle.x);
      const correctedRightKneeAlignment = Math.abs(correctedRightKnee.x - correctedRightAnkle.x);
      
      if (Math.abs(leftKneeAlignment - correctedLeftKneeAlignment) > 0.1 ||
          Math.abs(rightKneeAlignment - correctedRightKneeAlignment) > 0.1) {
        feedback.push('Keep your knees aligned with your toes');
        score -= 20;
      }
      
      // Check hip depth
      const hipDepth = (leftHip.y + rightHip.y) / 2;
      const correctedHipDepth = (correctedLeftHip.y + correctedRightHip.y) / 2;
      
      if (Math.abs(hipDepth - correctedHipDepth) > 0.1) {
        feedback.push('Squat deeper');
        score -= 20;
      }
      
      // Check back angle
      const backAngle = calculateAngle(leftHip, keypoints[11], keypoints[12]);
      const correctedBackAngle = calculateAngle(correctedLeftHip, correctedPose[5], correctedPose[8]);
      
      if (Math.abs(backAngle - correctedBackAngle) > 15) {
        feedback.push('Keep your back straight');
        score -= 20;
      }
      
      return {
        score: Math.max(0, score),
        feedback
      };
    }
  },
  '45° side bend': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftHip = keypoints[11];
      const rightHip = keypoints[12];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check hips if visible
      if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
        const hipLevel = Math.abs(leftHip.y - rightHip.y) < 0.1;
        checks.push(hipLevel);
        if (!hipLevel) feedback.push("Keep hips level");
      }
      
      // Check body angle if all points are visible
      if (leftShoulder && rightShoulder && leftHip && rightHip && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3) {
        const bodyAngle = calculateAngle(leftShoulder, leftHip, rightHip);
        if (bodyAngle !== null) {
          const correctAngle = Math.abs(bodyAngle - 45) < 10;
          checks.push(correctAngle);
          if (!correctAngle) feedback.push("Maintain 45-degree bend");
        }
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'air bike': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftHip = keypoints[11];
      const rightHip = keypoints[12];
      const leftKnee = keypoints[13];
      const rightKnee = keypoints[14];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check hips if visible
      if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
        const hipLevel = Math.abs(leftHip.y - rightHip.y) < 0.1;
        checks.push(hipLevel);
        if (!hipLevel) feedback.push("Keep hips level");
      }
      
      // Check opposite motion if all points are visible
      if (leftElbow && rightElbow && leftKnee && rightKnee && 
          leftShoulder && rightShoulder && leftHip && rightHip && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftKnee.score > 0.3 && rightKnee.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftHip);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightHip);
        const leftKneeAngle = calculateAngle(leftHip, leftKnee, rightHip);
        const rightKneeAngle = calculateAngle(rightHip, rightKnee, leftHip);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null && 
            leftKneeAngle !== null && rightKneeAngle !== null) {
          const oppositeMotion = Math.abs(leftElbowAngle - leftKneeAngle) > 90 && 
                                Math.abs(rightElbowAngle - rightKneeAngle) > 90;
          checks.push(oppositeMotion);
          if (!oppositeMotion) feedback.push("Move elbows and knees in opposite directions");
        }
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'walking on stepmill': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftHip = keypoints[11];
      const rightHip = keypoints[12];
      const leftKnee = keypoints[13];
      const rightKnee = keypoints[14];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check hips if visible
      if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
        const hipLevel = Math.abs(leftHip.y - rightHip.y) < 0.1;
        checks.push(hipLevel);
        if (!hipLevel) feedback.push("Keep hips level");
      }
      
      // Check alternate motion if knees are visible
      if (leftKnee && rightKnee && leftHip && rightHip && 
          leftKnee.score > 0.3 && rightKnee.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3) {
        const leftKneeAngle = calculateAngle(leftHip, leftKnee, rightHip);
        const rightKneeAngle = calculateAngle(rightHip, rightKnee, leftHip);
        
        if (leftKneeAngle !== null && rightKneeAngle !== null) {
          const alternateMotion = Math.abs(leftKneeAngle - rightKneeAngle) > 30;
          checks.push(alternateMotion);
          if (!alternateMotion) feedback.push("Step alternately with each leg");
        }
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell deadlift': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftHip = keypoints[11];
      const rightHip = keypoints[12];
      const leftKnee = keypoints[13];
      const rightKnee = keypoints[14];
      const leftAnkle = keypoints[15];
      const rightAnkle = keypoints[16];
      
      let checks = [];
      let feedback = [];
      
      // Check back if shoulders and hips are visible
      if (leftShoulder && rightShoulder && leftHip && rightHip && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3) {
        const backStraight = Math.abs(
          (leftShoulder.y + rightShoulder.y) / 2 - 
          (leftHip.y + rightHip.y) / 2
        ) < 0.2;
        checks.push(backStraight);
        if (!backStraight) feedback.push("Keep back straight");
      }
      
      // Check knee alignment if visible
      if (leftKnee && rightKnee && leftAnkle && rightAnkle && 
          leftKnee.score > 0.3 && rightKnee.score > 0.3 && 
          leftAnkle.score > 0.3 && rightAnkle.score > 0.3) {
        const leftKneeAlignment = Math.abs(leftKnee.x - leftAnkle.x) < 0.1;
        const rightKneeAlignment = Math.abs(rightKnee.x - rightAnkle.x) < 0.1;
        checks.push(leftKneeAlignment, rightKneeAlignment);
        if (!leftKneeAlignment) feedback.push("Align left knee with toes");
        if (!rightKneeAlignment) feedback.push("Align right knee with toes");
      }
      
      // Check hips if visible
      if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
        const hipLevel = Math.abs(leftHip.y - rightHip.y) < 0.1;
        checks.push(hipLevel);
        if (!hipLevel) feedback.push("Keep hips level");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell row': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftHip = keypoints[11];
      const rightHip = keypoints[12];
      
      let checks = [];
      let feedback = [];
      
      // Check back if shoulders and hips are visible
      if (leftShoulder && rightShoulder && leftHip && rightHip && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3) {
        const backStraight = Math.abs(
          (leftShoulder.y + rightShoulder.y) / 2 - 
          (leftHip.y + rightHip.y) / 2
        ) < 0.2;
        checks.push(backStraight);
        if (!backStraight) feedback.push("Keep back straight");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftHip && rightHip && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftHip);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightHip);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsClose = Math.abs(leftElbowAngle - 45) < 15 && Math.abs(rightElbowAngle - 45) < 15;
          checks.push(elbowsClose);
          if (!elbowsClose) feedback.push("Keep elbows close to body");
        }
      }
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell shoulder press': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowAngles = Math.abs(leftElbowAngle - 90) < 10 && Math.abs(rightElbowAngle - 90) < 10;
          checks.push(elbowAngles);
          if (!elbowAngles) feedback.push("Maintain 90-degree elbow angle");
        }
      }
      
      // Check wrists if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristAlignment = Math.abs(leftWrist.x - leftElbow.x) < 0.1;
        const rightWristAlignment = Math.abs(rightWrist.x - rightElbow.x) < 0.1;
        checks.push(leftWristAlignment, rightWristAlignment);
        if (!leftWristAlignment) feedback.push("Align left wrist above elbow");
        if (!rightWristAlignment) feedback.push("Align right wrist above elbow");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell shrug': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const straightArms = Math.abs(leftElbowAngle - 180) < 10 && Math.abs(rightElbowAngle - 180) < 10;
          checks.push(straightArms);
          if (!straightArms) feedback.push("Keep arms straight");
        }
      }
      
      // Check wrists if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristAlignment = Math.abs(leftWrist.x - leftElbow.x) < 0.1;
        const rightWristAlignment = Math.abs(rightWrist.x - rightElbow.x) < 0.1;
        checks.push(leftWristAlignment, rightWristAlignment);
        if (!leftWristAlignment) feedback.push("Align left wrist above elbow");
        if (!rightWristAlignment) feedback.push("Align right wrist above elbow");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell triceps extension': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowAngles = Math.abs(leftElbowAngle - 90) < 10 && Math.abs(rightElbowAngle - 90) < 10;
          checks.push(elbowAngles);
          if (!elbowAngles) feedback.push("Keep elbows at 90 degrees");
        }
      }
      
      // Check wrists if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristAlignment = Math.abs(leftWrist.x - leftElbow.x) < 0.1;
        const rightWristAlignment = Math.abs(rightWrist.x - rightElbow.x) < 0.1;
        checks.push(leftWristAlignment, rightWristAlignment);
        if (!leftWristAlignment) feedback.push("Align left wrist above elbow");
        if (!rightWristAlignment) feedback.push("Align right wrist above elbow");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell wrist curl': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsResting = Math.abs(leftElbowAngle - 90) < 5 && Math.abs(rightElbowAngle - 90) < 5;
          checks.push(elbowsResting);
          if (!elbowsResting) feedback.push("Keep elbows resting on surface");
        }
      }
      
      // Check wrist movement if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristMovement = Math.abs(leftWrist.y - leftElbow.y) > 0.1;
        const rightWristMovement = Math.abs(rightWrist.y - rightElbow.y) > 0.1;
        checks.push(leftWristMovement, rightWristMovement);
        if (!leftWristMovement) feedback.push("Move left wrist up and down");
        if (!rightWristMovement) feedback.push("Move right wrist up and down");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell wrist extension': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsResting = Math.abs(leftElbowAngle - 90) < 5 && Math.abs(rightElbowAngle - 90) < 5;
          checks.push(elbowsResting);
          if (!elbowsResting) feedback.push("Keep elbows resting on surface");
        }
      }
      
      // Check wrist movement if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristMovement = Math.abs(leftWrist.y - leftElbow.y) > 0.1;
        const rightWristMovement = Math.abs(rightWrist.y - rightElbow.y) > 0.1;
        checks.push(leftWristMovement, rightWristMovement);
        if (!leftWristMovement) feedback.push("Move left wrist up and down");
        if (!rightWristMovement) feedback.push("Move right wrist up and down");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell wrist flexion': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsResting = Math.abs(leftElbowAngle - 90) < 5 && Math.abs(rightElbowAngle - 90) < 5;
          checks.push(elbowsResting);
          if (!elbowsResting) feedback.push("Keep elbows resting on surface");
        }
      }
      
      // Check wrist movement if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristMovement = Math.abs(leftWrist.y - leftElbow.y) > 0.1;
        const rightWristMovement = Math.abs(rightWrist.y - rightElbow.y) > 0.1;
        checks.push(leftWristMovement, rightWristMovement);
        if (!leftWristMovement) feedback.push("Move left wrist up and down");
        if (!rightWristMovement) feedback.push("Move right wrist up and down");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell wrist pronation': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsResting = Math.abs(leftElbowAngle - 90) < 5 && Math.abs(rightElbowAngle - 90) < 5;
          checks.push(elbowsResting);
          if (!elbowsResting) feedback.push("Keep elbows resting on surface");
        }
      }
      
      // Check wrist rotation if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristRotation = Math.abs(leftWrist.x - leftElbow.x) > 0.1;
        const rightWristRotation = Math.abs(rightWrist.x - rightElbow.x) > 0.1;
        checks.push(leftWristRotation, rightWristRotation);
        if (!leftWristRotation) feedback.push("Rotate left wrist");
        if (!rightWristRotation) feedback.push("Rotate right wrist");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell wrist supination': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsResting = Math.abs(leftElbowAngle - 90) < 5 && Math.abs(rightElbowAngle - 90) < 5;
          checks.push(elbowsResting);
          if (!elbowsResting) feedback.push("Keep elbows resting on surface");
        }
      }
      
      // Check wrist rotation if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristRotation = Math.abs(leftWrist.x - leftElbow.x) > 0.1;
        const rightWristRotation = Math.abs(rightWrist.x - rightElbow.x) > 0.1;
        checks.push(leftWristRotation, rightWristRotation);
        if (!leftWristRotation) feedback.push("Rotate left wrist");
        if (!rightWristRotation) feedback.push("Rotate right wrist");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell wrist ulnar deviation': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsResting = Math.abs(leftElbowAngle - 90) < 5 && Math.abs(rightElbowAngle - 90) < 5;
          checks.push(elbowsResting);
          if (!elbowsResting) feedback.push("Keep elbows resting on surface");
        }
      }
      
      // Check wrist movement if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristMovement = Math.abs(leftWrist.x - leftElbow.x) > 0.1;
        const rightWristMovement = Math.abs(rightWrist.x - rightElbow.x) > 0.1;
        checks.push(leftWristMovement, rightWristMovement);
        if (!leftWristMovement) feedback.push("Move left wrist side to side");
        if (!rightWristMovement) feedback.push("Move right wrist side to side");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'barbell wrist radial deviation': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      const leftWrist = keypoints[9];
      const rightWrist = keypoints[10];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check elbows if visible
      if (leftElbow && rightElbow && leftShoulder && rightShoulder && leftWrist && rightWrist && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null) {
          const elbowsResting = Math.abs(leftElbowAngle - 90) < 5 && Math.abs(rightElbowAngle - 90) < 5;
          checks.push(elbowsResting);
          if (!elbowsResting) feedback.push("Keep elbows resting on surface");
        }
      }
      
      // Check wrist movement if visible
      if (leftWrist && rightWrist && leftElbow && rightElbow && 
          leftWrist.score > 0.3 && rightWrist.score > 0.3 && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3) {
        const leftWristMovement = Math.abs(leftWrist.x - leftElbow.x) > 0.1;
        const rightWristMovement = Math.abs(rightWrist.x - rightElbow.x) > 0.1;
        checks.push(leftWristMovement, rightWristMovement);
        if (!leftWristMovement) feedback.push("Move left wrist side to side");
        if (!rightWristMovement) feedback.push("Move right wrist side to side");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'running': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftHip = keypoints[11];
      const rightHip = keypoints[12];
      const leftKnee = keypoints[13];
      const rightKnee = keypoints[14];
      const leftAnkle = keypoints[15];
      const rightAnkle = keypoints[16];
      const leftElbow = keypoints[7];
      const rightElbow = keypoints[8];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check hips if visible
      if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
        const hipLevel = Math.abs(leftHip.y - rightHip.y) < 0.1;
        checks.push(hipLevel);
        if (!hipLevel) feedback.push("Keep hips level");
      }
      
      // Check knee motion if visible
      if (leftKnee && rightKnee && leftHip && rightHip && leftAnkle && rightAnkle && 
          leftKnee.score > 0.3 && rightKnee.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3 && 
          leftAnkle.score > 0.3 && rightAnkle.score > 0.3) {
        const leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
        const rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
        
        if (leftKneeAngle !== null && rightKneeAngle !== null) {
          const alternateMotion = Math.abs(leftKneeAngle - rightKneeAngle) > 30;
          checks.push(alternateMotion);
          if (!alternateMotion) feedback.push("Step alternately with each leg");
        }
      }
      
      // Check arm-leg coordination if visible
      if (leftElbow && rightElbow && leftKnee && rightKnee && 
          leftShoulder && rightShoulder && leftHip && rightHip && 
          leftElbow.score > 0.3 && rightElbow.score > 0.3 && 
          leftKnee.score > 0.3 && rightKnee.score > 0.3 && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftHip.score > 0.3 && rightHip.score > 0.3) {
        const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftHip);
        const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightHip);
        const leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
        const rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
        
        if (leftElbowAngle !== null && rightElbowAngle !== null && 
            leftKneeAngle !== null && rightKneeAngle !== null) {
          const armLegCoordination = Math.abs(leftElbowAngle - leftKneeAngle) > 90 && 
                                    Math.abs(rightElbowAngle - rightKneeAngle) > 90;
          checks.push(armLegCoordination);
          if (!armLegCoordination) feedback.push("Move arms alternately with legs");
        }
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  },
  'neck side stretch': {
    checkForm: (keypoints) => {
      const leftShoulder = keypoints[5];
      const rightShoulder = keypoints[6];
      const leftEar = keypoints[3];
      const rightEar = keypoints[4];
      
      let checks = [];
      let feedback = [];
      
      // Check shoulders if visible
      if (leftShoulder && rightShoulder && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y) < 0.1;
        checks.push(shoulderLevel);
        if (!shoulderLevel) feedback.push("Keep shoulders level");
      }
      
      // Check head tilt if ears are visible
      if (leftEar && rightEar && leftEar.score > 0.3 && rightEar.score > 0.3) {
        const headTilt = Math.abs(leftEar.y - rightEar.y) > 0.1;
        checks.push(headTilt);
        if (!headTilt) feedback.push("Tilt head to the side");
      }
      
      // Check shoulder relaxation if ears and shoulders are visible
      if (leftShoulder && rightShoulder && leftEar && rightEar && 
          leftShoulder.score > 0.3 && rightShoulder.score > 0.3 && 
          leftEar.score > 0.3 && rightEar.score > 0.3) {
        const shouldersRelaxed = Math.abs(leftShoulder.y - leftEar.y) > 0.2 && 
                                Math.abs(rightShoulder.y - rightEar.y) > 0.2;
        checks.push(shouldersRelaxed);
        if (!shouldersRelaxed) feedback.push("Keep shoulders relaxed");
      }
      
      return {
        score: checks.length > 0 ? (checks.filter(Boolean).length / checks.length) * 10 : 0,
        feedback: feedback
      };
    }
  }
};

function calculateAngle(p1, p2, p3) {
  const v1 = {
    x: p1.x - p2.x,
    y: p1.y - p2.y
  };
  const v2 = {
    x: p3.x - p2.x,
    y: p3.y - p2.y
  };
  
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
  
  return Math.acos(dot / (mag1 * mag2)) * (180 / Math.PI);
}

function populateExerciseList() {
  exerciseLabels.forEach(label => {
    const option = document.createElement("option");
    option.value = label;
    exerciseListDatalist.appendChild(option);
  });

  exerciseSelectInput.addEventListener("input", () => {
    selectedExercise = exerciseSelectInput.value;
    workoutNameInput.value = selectedExercise;
  });
}

function normalizeKeypoints(keypoints) {
  // Center the pose around the hip center
  const leftHip = keypoints[11];
  const rightHip = keypoints[12];
  const hipCenter = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2
  };
  
  // Calculate scale using multiple body measurements for better stability
  const leftShoulder = keypoints[5];
  const rightShoulder = keypoints[6];
  const shoulderWidth = Math.sqrt(
    Math.pow(rightShoulder.x - leftShoulder.x, 2) +
    Math.pow(rightShoulder.y - leftShoulder.y, 2)
  );
  
  const leftKnee = keypoints[13];
  const rightKnee = keypoints[14];
  const kneeWidth = Math.sqrt(
    Math.pow(rightKnee.x - leftKnee.x, 2) +
    Math.pow(rightKnee.y - leftKnee.y, 2)
  );
  
  // Use average of shoulder and knee width for more stable scaling
  const scale = (shoulderWidth + kneeWidth) / 2;
  
  // Normalize each keypoint with improved scaling
  return keypoints.map(kp => ({
    x: (kp.x - hipCenter.x) / scale,
    y: (kp.y - hipCenter.y) / scale,
    score: kp.score
  }));
}

function smoothKeypoints(keypoints, prevKeypoints) {
  if (!prevKeypoints) return keypoints;
  
  const smoothingFactor = 0.4; // Increased smoothing for more stability
  
  return keypoints.map((kp, i) => {
    if (kp.score < 0.3) return kp; // Don't smooth low confidence points
    
    return {
      x: kp.x * (1 - smoothingFactor) + prevKeypoints[i].x * smoothingFactor,
      y: kp.y * (1 - smoothingFactor) + prevKeypoints[i].y * smoothingFactor,
      score: kp.score
    };
  });
}

let prevKeypoints = null;

let repCount = 0;
let lastRepTime = 0;
const REP_COOLDOWN = 2000; // Minimum time between reps (2 seconds)
const REP_THRESHOLD = 0.7; // Threshold for detecting a rep

// Add GIF overlay element with improved styling
const gifOverlay = document.createElement("img");
gifOverlay.style.position = "absolute";
gifOverlay.style.top = "20px";
gifOverlay.style.right = "20px";
gifOverlay.style.width = "200px";
gifOverlay.style.height = "auto";
gifOverlay.style.zIndex = "1000";
gifOverlay.style.display = "none";
gifOverlay.style.borderRadius = "10px";
gifOverlay.style.boxShadow = "0 0 10px rgba(0,0,0,0.5)";
document.querySelector(".webcam-section").appendChild(gifOverlay);

// Add rep counter display with improved styling
const repCounterDisplay = document.createElement("div");
repCounterDisplay.style.position = "absolute";
repCounterDisplay.style.top = "20px";
repCounterDisplay.style.left = "20px";
repCounterDisplay.style.backgroundColor = "rgba(0, 0, 0, 0.8)";
repCounterDisplay.style.color = "#00ff00";
repCounterDisplay.style.padding = "15px";
repCounterDisplay.style.borderRadius = "10px";
repCounterDisplay.style.zIndex = "1000";
repCounterDisplay.style.fontSize = "24px";
repCounterDisplay.style.fontWeight = "bold";
repCounterDisplay.style.fontFamily = "Arial, sans-serif";
repCounterDisplay.style.boxShadow = "0 0 10px rgba(0,255,0,0.5)";
repCounterDisplay.textContent = "Reps: 0";
document.querySelector(".webcam-section").appendChild(repCounterDisplay);

// Store rep counts in localStorage
let repCounts = JSON.parse(localStorage.getItem("repCounts")) || {};

// Update rep counter display
function updateRepCounterDisplay() {
  const currentExercise = workoutNameInput.value.toLowerCase();
  const state = repStates[currentExercise];
  if (state) {
    repCounts[currentExercise] = state.count;
    localStorage.setItem("repCounts", JSON.stringify(repCounts));
    repCounterDisplay.textContent = `Reps: ${state.count}`;
  }
}

// Enhanced rep counting state with more precise thresholds and support for more exercises
const repStates = {
  'barbell bench press': {
    lastPosition: null,
    direction: 'up',
    threshold: 0.15,
    count: 0,
    minAngle: 45,
    maxAngle: 90,
    keypoints: {
      shoulder: [5, 6],
      elbow: [7, 8],
      wrist: [9, 10]
    }
  },
  'barbell squat': {
    lastPosition: null,
    direction: 'down',
    threshold: 0.2,
    count: 0,
    minAngle: 70,
    maxAngle: 100,
    keypoints: {
      hip: [11, 12],
      knee: [13, 14],
      ankle: [15, 16]
    }
  },
  'push-up': {
    lastPosition: null,
    direction: 'down',
    threshold: 0.15,
    count: 0,
    minAngle: 45,
    maxAngle: 90,
    keypoints: {
      shoulder: [5, 6],
      elbow: [7, 8],
      wrist: [9, 10]
    }
  },
  'pull-up': {
    lastPosition: null,
    direction: 'up',
    threshold: 0.15,
    count: 0,
    minAngle: 45,
    maxAngle: 90,
    keypoints: {
      shoulder: [5, 6],
      elbow: [7, 8],
      wrist: [9, 10]
    }
  },
  'deadlift': {
    lastPosition: null,
    direction: 'down',
    threshold: 0.2,
    count: 0,
    minAngle: 70,
    maxAngle: 100,
    keypoints: {
      hip: [11, 12],
      knee: [13, 14],
      ankle: [15, 16]
    }
  },
  'shoulder press': {
    lastPosition: null,
    direction: 'up',
    threshold: 0.15,
    count: 0,
    minAngle: 45,
    maxAngle: 90,
    keypoints: {
      shoulder: [5, 6],
      elbow: [7, 8],
      wrist: [9, 10]
    }
  },
  'neck side stretch': {
    lastPosition: null,
    direction: 'left',
    threshold: 0.05,
    count: 0,
    minAngle: 15,
    maxAngle: 40,
    keypoints: {
      ear: [3, 4],
      shoulder: [5, 6]
    },
    movementType: 'side',
    lastAngle: null,
    angleHistory: [],
    historySize: 10,
    peakAngle: null,
    valleyAngle: null,
    repInProgress: false
  }
};

function checkRep(keypoints, exercise) {
  if (!repStates[exercise]) return;
  
  const state = repStates[exercise];
  const now = Date.now();
  
  // Get relevant keypoints based on exercise
  const kps = state.keypoints;
  let currentPosition;
  let currentAngle;
  
  // Calculate average position and angle for the exercise
  if (kps.shoulder && kps.elbow && kps.wrist) {
    // For upper body exercises
    const leftElbowAngle = calculateAngle(
      keypoints[kps.shoulder[0]],
      keypoints[kps.elbow[0]],
      keypoints[kps.wrist[0]]
    );
    const rightElbowAngle = calculateAngle(
      keypoints[kps.shoulder[1]],
      keypoints[kps.elbow[1]],
      keypoints[kps.wrist[1]]
    );
    currentAngle = (leftElbowAngle + rightElbowAngle) / 2;
    currentPosition = (keypoints[kps.wrist[0]].y + keypoints[kps.wrist[1]].y) / 2;
  } else if (kps.hip && kps.knee && kps.ankle) {
    // For lower body exercises
    const leftKneeAngle = calculateAngle(
      keypoints[kps.hip[0]],
      keypoints[kps.knee[0]],
      keypoints[kps.ankle[0]]
    );
    const rightKneeAngle = calculateAngle(
      keypoints[kps.hip[1]],
      keypoints[kps.knee[1]],
      keypoints[kps.ankle[1]]
    );
    currentAngle = (leftKneeAngle + rightKneeAngle) / 2;
    currentPosition = (keypoints[kps.hip[0]].y + keypoints[kps.hip[1]].y) / 2;
  } else if (kps.ear && kps.shoulder) {
    // For neck side stretch
    const leftNeckAngle = calculateAngle(
      keypoints[kps.ear[0]],
      keypoints[kps.shoulder[0]],
      keypoints[kps.shoulder[1]]
    );
    const rightNeckAngle = calculateAngle(
      keypoints[kps.ear[1]],
      keypoints[kps.shoulder[1]],
      keypoints[kps.shoulder[0]]
    );
    currentAngle = (leftNeckAngle + rightNeckAngle) / 2;
    currentPosition = (keypoints[kps.ear[0]].x + keypoints[kps.ear[1]].x) / 2;
    
    // Add to angle history for neck side stretch
    if (state.angleHistory) {
      state.angleHistory.push(currentAngle);
      if (state.angleHistory.length > state.historySize) {
        state.angleHistory.shift();
      }
    }
  }
  
  if (state.lastPosition === null || currentAngle === null) {
    state.lastPosition = currentPosition;
    state.lastAngle = currentAngle;
    return;
  }
  
  const movement = currentPosition - state.lastPosition;
  const angleChange = currentAngle - state.lastAngle;
  
  // Enhanced rep detection logic with hysteresis and support for side movements
  let repCompleted = false;
  
  if (state.movementType === 'side') {
    // Special handling for neck side stretch
    if (exercise === 'neck side stretch') {
      // Initialize peak and valley angles if not set
      if (state.peakAngle === null) {
        state.peakAngle = currentAngle;
        state.valleyAngle = currentAngle;
      }
      
      // Update peak and valley angles
      if (currentAngle > state.peakAngle) {
        state.peakAngle = currentAngle;
      }
      if (currentAngle < state.valleyAngle) {
        state.valleyAngle = currentAngle;
      }
      
      // Check if we have enough history
      if (state.angleHistory && state.angleHistory.length >= 5) {
        const recentAngles = state.angleHistory.slice(-5);
        
        // Detect if we're at a peak (angle is decreasing after increasing)
        const isAtPeak = recentAngles[4] < recentAngles[3] && 
                         recentAngles[3] < recentAngles[2] && 
                         recentAngles[2] > recentAngles[1] && 
                         recentAngles[1] > recentAngles[0];
        
        // Detect if we're at a valley (angle is increasing after decreasing)
        const isAtValley = recentAngles[4] > recentAngles[3] && 
                           recentAngles[3] > recentAngles[2] && 
                           recentAngles[2] < recentAngles[1] && 
                           recentAngles[1] < recentAngles[0];
        
        // If we're at a peak and the angle is significant, mark a rep in progress
        if (isAtPeak && state.peakAngle - state.valleyAngle > 10 && !state.repInProgress) {
          state.repInProgress = true;
          console.log("Rep started at peak angle:", state.peakAngle);
        }
        
        // If we're at a valley and a rep was in progress, complete the rep
        if (isAtValley && state.repInProgress) {
          repCompleted = true;
          state.repInProgress = false;
          console.log("Rep completed at valley angle:", state.valleyAngle);
          
          // Reset peak and valley for next rep
          state.peakAngle = currentAngle;
          state.valleyAngle = currentAngle;
        }
      }
    } else {
      // For other side-to-side movements
      if (state.direction === 'left') {
        if (currentAngle >= state.maxAngle && Math.abs(angleChange) > state.threshold) {
          repCompleted = true;
          state.direction = 'right';
        }
      } else {
        if (currentAngle <= state.minAngle && Math.abs(angleChange) > state.threshold) {
          state.direction = 'left';
        }
      }
    }
  } else {
    // For up/down movements
    if (state.direction === 'down') {
      if (currentAngle <= state.minAngle && Math.abs(angleChange) > state.threshold) {
        repCompleted = true;
        state.direction = 'up';
      }
    } else {
      if (currentAngle >= state.maxAngle && Math.abs(angleChange) > state.threshold) {
        state.direction = 'down';
      }
    }
  }
  
  // Update state
  state.lastPosition = currentPosition;
  state.lastAngle = currentAngle;
  
  // Increment rep count if completed
  if (repCompleted && (now - lastRepTime) > REP_COOLDOWN) {
    state.count++;
    lastRepTime = now;
    updateRepCounterDisplay();
    
    // Visual feedback
    repCounterDisplay.style.transform = 'scale(1.2)';
    setTimeout(() => {
      repCounterDisplay.style.transform = 'scale(1)';
    }, 200);
  }
}

function checkKeypointVisibility(keypoints) {
  // Count visible keypoints (score > 0.3)
  const visibleKeypoints = keypoints.filter(kp => kp.score > 0.3).length;
  const totalKeypoints = keypoints.length;
  const visibilityRatio = visibleKeypoints / totalKeypoints;
  
  return {
    visible: visibilityRatio >= 0.5,
    feedback: visibilityRatio >= 0.5 ? "✅ Ready" : "⚠️ Adjust position"
  };
}

async function loadModels() {
  try {
    console.log("Starting to load models...");
    await tf.setBackend('webgl');
    await tf.ready();

    console.log("Loading MoveNet model...");
    poseDetector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
        enableSmoothing: true,
        minPoseScore: 0.25
      }
    );
    console.log("✅ MoveNet loaded!");

    console.log("Loading exercise labels...");
    const labelsRes = await fetch("/tfjs_model/exercise_labels.json");
    if (!labelsRes.ok) {
      throw new Error(`Failed to load exercise labels: ${labelsRes.status} ${labelsRes.statusText}`);
    }
    exerciseLabels = await labelsRes.json();
    console.log("✅ Labels loaded:", exerciseLabels);

    populateExerciseList();

    classifierModel = await tf.loadLayersModel("/tfjs_model/model.json");
    console.log("✅ Classifier model loaded!");

    predictionBox.textContent = "✅ Models loaded!";
    startCamera();
  } catch (err) {
    console.error("❌ Error loading models:", err);
    predictionBox.textContent = `❌ Error: ${err.message}`;
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      }
    });
    videoElement.srcObject = stream;
    await videoElement.play();

    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    videoElement.style.display = "block";
    videoElement.style.width = "100%";
    videoElement.style.maxWidth = "640px";

    predictPoses();
  } catch (err) {
    console.error("❌ Camera error:", err);
    predictionBox.textContent = `❌ Camera error: ${err.message}`;
  }
}

function smoothPredictions(newPredictions) {
  lastPredictions.push(newPredictions);
  if (lastPredictions.length > SMOOTHING_WINDOW_SIZE) {
    lastPredictions.shift();
  }

  const avgPredictions = new Array(newPredictions.length).fill(0);
  for (const predictions of lastPredictions) {
    for (let i = 0; i < predictions.length; i++) {
      avgPredictions[i] += predictions[i] / lastPredictions.length;
    }
  }

  return avgPredictions;
}

async function predictPoses() {
  if (!videoElement.videoWidth) {
    requestAnimationFrame(predictPoses);
    return;
  }

  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  ctx.drawImage(videoElement, 0, 0);

  try {
    const poses = await poseDetector.estimatePoses(videoElement);

    if (poses.length > 0) {
      const pose = poses[0];

      if (pose.score > MIN_POSE_SCORE) {
        const smoothedKeypoints = smoothKeypoints(pose.keypoints, prevKeypoints);
        prevKeypoints = smoothedKeypoints;
        
        const normalizedKeypoints = normalizeKeypoints(smoothedKeypoints);
        
        drawKeypoints(pose.keypoints);
        drawConnectors(pose.keypoints);

        const selectedExercise = workoutNameInput.value.toLowerCase();
        
        if (FORM_RULES[selectedExercise]) {
          const formCheck = FORM_RULES[selectedExercise].checkForm(normalizedKeypoints);
          
          // Check for reps
          checkRep(normalizedKeypoints, selectedExercise);
          
          // Show exercise GIF
          gifOverlay.style.display = "block";
          gifOverlay.src = `/gifs/${selectedExercise.replace(/\s+/g, '_')}.gif`;
          
          // Combine feedback
          let feedback = [];
          
          if (formCheck.feedback.length > 0) {
            feedback.push(...formCheck.feedback);
          } else {
            feedback.push("✅ Good form");
          }
          
          predictionBox.textContent = feedback.join('\n');
        } else {
          predictionBox.textContent = "🤔 Select an exercise to check form";
          gifOverlay.style.display = "none";
        }
      } else {
        predictionBox.textContent = "⚠️ Adjust position";
        gifOverlay.style.display = "none";
      }
    }
  } catch (err) {
    console.error("Error in pose detection:", err);
  }

  requestAnimationFrame(predictPoses);
}

function drawKeypoints(keypoints) {
  ctx.fillStyle = "#FF0000";
  for (const kp of keypoints) {
    if (kp.score > 0.3) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function drawConnectors(keypoints) {
  const connections = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]
  ];

  ctx.strokeStyle = "#00FF00";
  ctx.lineWidth = 2;

  for (const [i, j] of connections) {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];
    if (kp1.score > 0.3 && kp2.score > 0.3) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  }
}

const logForm = document.getElementById("logForm");
if (logForm) {
  logForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const workoutName = workoutNameInput.value;
  const formScore = formScoreInput.value;
  const date = document.getElementById("date").value;
  const token = localStorage.getItem("token");

  try {
    const res = await fetch("http://localhost:5000/history", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ workoutName, formScore, date }),
    });

    const data = await res.json();
    if (res.ok && data.message === "Workout logged successfully!") {
      alert(data.message);
      window.location.href = "history.html";
    } else {
      alert(data.message || "Failed to log workout.");
    }
  } catch (err) {
      alert("Error: " + err.message);
    }
  });
}

// Update reset button to clear specific exercise reps
const resetButton = document.createElement("button");
resetButton.textContent = "Reset Reps";
resetButton.style.marginTop = "10px";
resetButton.style.padding = "10px 20px";
resetButton.style.backgroundColor = "#ff4444";
resetButton.style.color = "white";
resetButton.style.border = "none";
resetButton.style.borderRadius = "5px";
resetButton.style.cursor = "pointer";
resetButton.onclick = () => {
  const currentExercise = workoutNameInput.value.toLowerCase();
  if (repStates[currentExercise]) {
    repStates[currentExercise].count = 0;
    repStates[currentExercise].lastPosition = null;
    repStates[currentExercise].direction = 'up';
    updateRepCounterDisplay();
  }
};
document.querySelector(".webcam-section").appendChild(resetButton);

// Update exercise selection to show correct rep count and GIF
exerciseSelectInput.addEventListener("input", () => {
  selectedExercise = exerciseSelectInput.value;
  workoutNameInput.value = selectedExercise;
  updateRepCounterDisplay();
  
  // Update GIF
  if (FORM_RULES[selectedExercise.toLowerCase()]) {
    gifOverlay.style.display = "block";
    gifOverlay.src = `/gifs/${selectedExercise.toLowerCase().replace(/\s+/g, '_')}.gif`;
  } else {
    gifOverlay.style.display = "none";
  }
});

// Add new function for pose correction
async function correctPose(keypoints) {
    try {
        if (!selectedExercise) {
            console.warn('No exercise selected');
            return null;
        }

        // Convert keypoints to the format expected by the model
        const normalizedPose = normalizePose(convertToH36MFormat(keypoints));
        
        // Send pose to server for correction
        const response = await fetch('/api/correct-pose', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pose: normalizedPose,
                exercise: selectedExercise
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Update exercise confidence
        const confidence = data.exerciseConfidence[0];
        if (confidence > CONFIDENCE_THRESHOLD) {
            updateExerciseConfidence(confidence);
        }

        // Convert corrected pose back to keypoint format
        const correctedKeypoints = data.correctedPose.map((point, i) => ({
            x: point[0],
            y: point[1],
            z: point[2],
            score: 1.0
        }));

        return correctedKeypoints;
    } catch (error) {
        console.error('Error correcting pose:', error);
        return null;
    }
}

function updateExerciseConfidence(confidence) {
    const confidenceElement = document.getElementById('exerciseConfidence');
    if (confidenceElement) {
        confidenceElement.textContent = `Exercise Confidence: ${(confidence * 100).toFixed(1)}%`;
        confidenceElement.style.color = confidence > 0.8 ? 'green' : confidence > 0.5 ? 'orange' : 'red';
    }
}

function convertToH36MFormat(keypoints) {
  // Map MediaPipe keypoints to H36M format
  // This is a simplified mapping - you'll need to adjust based on your needs
  const h36mJoints = 17;
  const h36mPose = new Array(h36mJoints).fill(null);
  
  // Example mapping (adjust based on your needs):
  // H36M: 0=hip, 1=spine, 2=chest, 3=neck, 4=head, 5=left shoulder, 6=left elbow, 7=left wrist,
  // 8=right shoulder, 9=right elbow, 10=right wrist, 11=left hip, 12=left knee, 13=left ankle,
  // 14=right hip, 15=right knee, 16=right ankle
  
  // Map MediaPipe keypoints to H36M format
  if (keypoints[23]) h36mPose[0] = keypoints[23];  // hip
  if (keypoints[11]) h36mPose[5] = keypoints[11];  // left shoulder
  if (keypoints[13]) h36mPose[6] = keypoints[13];  // left elbow
  if (keypoints[15]) h36mPose[7] = keypoints[15];  // left wrist
  if (keypoints[12]) h36mPose[8] = keypoints[12];  // right shoulder
  if (keypoints[14]) h36mPose[9] = keypoints[14];  // right elbow
  if (keypoints[16]) h36mPose[10] = keypoints[16]; // right wrist
  if (keypoints[23]) h36mPose[11] = keypoints[23]; // left hip
  if (keypoints[25]) h36mPose[12] = keypoints[25]; // left knee
  if (keypoints[27]) h36mPose[13] = keypoints[27]; // left ankle
  if (keypoints[24]) h36mPose[14] = keypoints[24]; // right hip
  if (keypoints[26]) h36mPose[15] = keypoints[26]; // right knee
  if (keypoints[28]) h36mPose[16] = keypoints[28]; // right ankle
  
  // Fill in missing joints with interpolated values
  for (let i = 0; i < h36mJoints; i++) {
    if (!h36mPose[i]) {
      // Find the closest non-null joints
      let leftIdx = i - 1;
      let rightIdx = i + 1;
      
      while (leftIdx >= 0 && !h36mPose[leftIdx]) leftIdx--;
      while (rightIdx < h36mJoints && !h36mPose[rightIdx]) rightIdx++;
      
      if (leftIdx >= 0 && rightIdx < h36mJoints) {
        // Interpolate between the two points
        const leftPoint = h36mPose[leftIdx];
        const rightPoint = h36mPose[rightIdx];
        const ratio = (i - leftIdx) / (rightIdx - leftIdx);
        
        h36mPose[i] = {
          x: leftPoint.x + (rightPoint.x - leftPoint.x) * ratio,
          y: leftPoint.y + (rightPoint.y - leftPoint.y) * ratio,
          z: leftPoint.z + (rightPoint.z - leftPoint.z) * ratio,
          score: Math.min(leftPoint.score, rightPoint.score)
        };
      } else if (leftIdx >= 0) {
        // Use the left point
        h36mPose[i] = {...h36mPose[leftIdx]};
      } else if (rightIdx < h36mJoints) {
        // Use the right point
        h36mPose[i] = {...h36mPose[rightIdx]};
      } else {
        // Default to origin if no points are available
        h36mPose[i] = {x: 0, y: 0, z: 0, score: 0};
      }
    }
  }
  
  return h36mPose;
}

function normalizePose(pose) {
  // Center around hip
  const hip = pose[0];
  if (!hip) return pose;
  
  const centeredPose = pose.map(point => {
    if (!point) return null;
    return {
      x: point.x - hip.x,
      y: point.y - hip.y,
      z: point.z - hip.z,
      score: point.score
    };
  });
  
  // Scale to unit size
  let maxDist = 0;
  centeredPose.forEach(point => {
    if (!point) return;
    const dist = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    maxDist = Math.max(maxDist, dist);
  });
  
  if (maxDist > 0) {
    return centeredPose.map(point => {
      if (!point) return null;
      return {
        x: point.x / maxDist,
        y: point.y / maxDist,
        z: point.z / maxDist,
        score: point.score
      };
    });
  }
  
  return centeredPose;
}

// Add a function to visualize the corrected pose
function visualizeCorrectedPose(ctx, originalPose, correctedPose, canvasWidth, canvasHeight) {
  // Draw the original pose in red
  drawPose(ctx, originalPose, canvasWidth, canvasHeight, 'rgba(255, 0, 0, 0.5)');
  
  // Draw the corrected pose in green
  drawPose(ctx, correctedPose, canvasWidth, canvasHeight, 'rgba(0, 255, 0, 0.5)');
  
  // Add a legend
  ctx.fillStyle = 'white';
  ctx.fillRect(10, 10, 200, 50);
  ctx.fillStyle = 'red';
  ctx.fillRect(20, 20, 20, 20);
  ctx.fillStyle = 'black';
  ctx.font = '14px Arial';
  ctx.fillText('Your Pose', 50, 35);
  
  ctx.fillStyle = 'green';
  ctx.fillRect(20, 40, 20, 20);
  ctx.fillStyle = 'black';
  ctx.fillText('Corrected Pose', 50, 55);
}

function drawPose(ctx, pose, canvasWidth, canvasHeight, color) {
  // Define the connections between joints
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],  // Spine
    [5, 6], [6, 7],  // Left arm
    [8, 9], [9, 10],  // Right arm
    [11, 12], [12, 13],  // Left leg
    [14, 15], [15, 16]  // Right leg
  ];
  
  // Draw connections
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  
  for (const [i, j] of connections) {
    const point1 = pose[i];
    const point2 = pose[j];
    
    if (point1 && point2) {
      // Scale and translate to canvas coordinates
      const x1 = (point1.x + 1) * canvasWidth / 2;
      const y1 = (1 - point1.y) * canvasHeight / 2;
      const x2 = (point2.x + 1) * canvasWidth / 2;
      const y2 = (1 - point2.y) * canvasHeight / 2;
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
  
  // Draw joints
  ctx.fillStyle = color;
  
  for (const point of pose) {
    if (point) {
      const x = (point.x + 1) * canvasWidth / 2;
      const y = (1 - point.y) * canvasHeight / 2;
      
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// Modify the existing onResults function to use pose correction
async function onResults(results) {
  if (results.poseLandmarks) {
    const keypoints = results.poseLandmarks;
    
    // Get pose correction if an exercise is selected
    if (selectedExercise) {
      const correctedPose = await correctPose(keypoints);
      if (correctedPose) {
        // Visualize both the original and corrected poses
        visualizeCorrectedPose(ctx, keypoints, correctedPose, canvasElement.width, canvasElement.height);
        
        // Calculate form score using both original and corrected poses
        const formResult = FORM_RULES[selectedExercise].checkForm(keypoints, correctedPose);
        formScoreInput.value = formResult.score.toFixed(1);
        
        // Update feedback
        const feedbackElement = document.getElementById('feedback');
        if (feedbackElement) {
          feedbackElement.innerHTML = formResult.feedback.join('<br>');
        }
      } else {
        // If pose correction failed, just draw the original pose
        drawKeypoints(keypoints);
        drawConnectors(keypoints);
      }
    } else {
      // If no exercise is selected, just draw the original pose
      drawKeypoints(keypoints);
      drawConnectors(keypoints);
    }
  }
  
  // Check for reps
  if (selectedExercise) {
    checkRep(keypoints, selectedExercise);
  }
}

loadModels();
