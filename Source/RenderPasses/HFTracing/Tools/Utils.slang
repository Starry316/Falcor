float MISWeight(float p0, float p1) {
    float q0 = p0 * p0;
    float q1 = p1 * p1;
    return q0 / (q0 + q1);
}
